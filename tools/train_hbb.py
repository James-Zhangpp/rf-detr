import os
import glob
from PIL import Image
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge

class YoloHBBTrainer:
    """
    使用 rf-detr 训练标准水平框 (HBB) 的工具类。
    底层使用 RFDETR 框架，并加载 YOLO 格式的数据集。
    支持 direct_resize (直接拉伸) 与 letterbox (保持长宽比缩放 + 灰度填充)。
    """
    def __init__(self, model_size: str = "rfdetr-medium", dataset_dir: str = ""):
        """
        初始化训练类
        
        Args:
            model_size (str): 模型大小，推荐 "rfdetr-medium", "rfdetr-small", "rfdetr-nano", "rfdetr-large" 等
            dataset_dir (str): YOLO 格式数据集所在的目录（需包含 data.yaml）
        """
        self.model_size = model_size
        self.dataset_dir = dataset_dir
        model_classes = {
            "rfdetr-nano": RFDETRNano,
            "rfdetr-small": RFDETRSmall,
            "rfdetr-medium": RFDETRMedium,
            "rfdetr-large": RFDETRLarge,
        }
        if model_size not in model_classes:
            raise ValueError(f"不支持的模型大小: {model_size}，请选择 {list(model_classes.keys())} 之一")
        self.model = model_classes[model_size]()
    
    def auto_optimize_params(self, target_img_size: int = 800, target_model_size: str = "rfdetr-medium"):
        """
        自动扫描数据集并针对目标分辨率（默认 800）与模型架构优化训练参数
        
        Args:
            target_img_size (int): 目标图像分辨率（默认 800）
            target_model_size (str): 目标模型大小（默认 rfdetr-medium）
        """
        print(f"[*] 开始扫描数据集进行训练参数优化 (目标模型: {target_model_size}, 目标分辨率: {target_img_size})...")
        # 1. 查找图片数量
        base_dir = Path(self.dataset_dir)
        img_paths = []
        for ext in ["*.jpg", "*.png", "*.bmp"]:
            img_paths.extend(base_dir.rglob(f"train/**/{ext}"))
            img_paths.extend(base_dir.rglob(f"images/train/**/{ext}"))
            if not img_paths:
                img_paths.extend(base_dir.rglob(ext))
        
        img_paths = list(set(img_paths))  # 去重
        num_images = len(img_paths)
        
        # 保证分辨率满足 patch_size(16) * num_windows(2) = 32 整除要求
        block_size = 32
        if target_img_size % block_size != 0:
            target_img_size = round(target_img_size / block_size) * block_size
            print(f"[!] 调整分辨率以满足 block_size={block_size} 整除约束: {target_img_size}")

        if num_images == 0:
            print(f"[!] 在 {base_dir} 中没有找到任何图片，采用保底参数。")
            return target_img_size, 220, 2, 4, 176, 3.0, target_model_size, 2
            
        # 2. 读取部分图片计算平均最大边长
        sample_count = min(30, num_images)
        max_dims = []
        for p in img_paths[:sample_count]:
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    max_dims.append(max(w, h))
            except Exception:
                pass
                
        avg_dim = sum(max_dims) / len(max_dims) if max_dims else float(target_img_size)
        img_size = target_img_size
            
        # 3. 决定 epochs (根据数据集规模动态规划收敛轮数)
        if num_images < 100:
            epochs = 300
        elif num_images < 500:
            epochs = 220
        else:
            epochs = 150
            
        # 4. 决定 batch_size 与梯度累加
        # 针对 800 分辨率与 medium 模型：batch_size=2, grad_accum=4（等效 Batch Size = 8，防止 OOM）
        # 针对 <= 640 分辨率或 small/nano 模型：batch_size=4, grad_accum=2
        if img_size >= 800 or "medium" in target_model_size or "large" in target_model_size:
            batch_size = 2
            grad_accum_steps = 4  # 2 * 4 = 8
        else:
            batch_size = 4
            grad_accum_steps = 2  # 4 * 2 = 8
            
        # 5. 学习率下降点（前 80% 保持主学习率）
        lr_drop = int(epochs * 0.8)
        warmup_epochs = 3.0
        rec_model_size = target_model_size
            
        # 6. 验证间隔（2 轮验证一次）
        eval_interval = 2
            
        print(f"[*] 数据分析完成: 训练集包含 {num_images} 张图片, 平均边长 {avg_dim:.1f}px")
        print(f"[*] 推荐训练参数 -> model_size: {rec_model_size}, img_size: {img_size}, epochs: {epochs}, "
              f"batch_size: {batch_size}, grad_accum: {grad_accum_steps}, lr_drop: {lr_drop}, warmup: {warmup_epochs}")
        return img_size, epochs, batch_size, grad_accum_steps, lr_drop, warmup_epochs, rec_model_size, eval_interval

    def train(self, epochs: int = 220, batch_size: int = 2, grad_accum_steps: int = 4,
              output_dir: str = "output", device: str = "auto", img_size: int = 800, 
              num_workers: int = 4, early_stopping: bool = False, auto_optimize: bool = True, 
              eval_interval: int = 2, multi_scale: bool = False, resize_mode: str = "letterbox",
              scale_jitter: bool = False, aug_config: Optional[Dict[str, Any]] = None):
        """
        启动训练
        
        Args:
            epochs (int): 训练的轮数（默认 220）
            batch_size (int): 批次大小（默认 2，800 分辨率推荐）
            grad_accum_steps (int): 梯度累加步数（默认 4，等效 batch size = 8）
            output_dir (str): 模型权重和日志的保存目录
            device (str): 训练设备，"auto" 表示自动检测，"cuda" 强制 GPU，"cpu" 强制 CPU
            img_size (int): 输入图像的尺寸（默认 800）
            num_workers (int): 数据加载的线程数（默认 4）
            early_stopping (bool): 是否开启早停
            auto_optimize (bool): 是否开启自动参数优化（基于数据集统计）
            eval_interval (int): 多少轮进行一次验证评估（默认 2）
            multi_scale (bool): 是否开启多尺度训练（False 为固定单尺度）
            resize_mode (str): 图像缩放模式，"letterbox" (保持长宽比+灰边填充) 或 "direct_resize" (直接拉伸)
            scale_jitter (bool): 是否开启随机裁剪缩放增强（False 为不裁剪）
            aug_config (dict): 自定义 Albumentations 数据增强配置（若为 None 且 resize_mode="letterbox" 则自动配置 Letterbox 管道）
        """
        # 自动提取数据集文件夹的名称作为项目名，并在输出目录下加一层子目录
        project_name = Path(self.dataset_dir).name
        output_dir = str(Path(output_dir) / project_name)
        
        lr_drop = int(epochs * 0.8)
        warmup_epochs = 3.0
        
        if auto_optimize:
            (opt_img_size, opt_epochs, opt_batch, opt_grad_accum, 
             opt_lr_drop, opt_warmup, opt_model_size, opt_eval_interval) = self.auto_optimize_params(
                 target_img_size=img_size, target_model_size=self.model_size
             )
            if opt_img_size is not None:
                img_size = opt_img_size
                epochs = opt_epochs
                batch_size = opt_batch
                grad_accum_steps = opt_grad_accum
                lr_drop = opt_lr_drop
                warmup_epochs = opt_warmup
                eval_interval = opt_eval_interval
                if opt_model_size != self.model_size:
                    print(f"[*] 自动优化将模型从 {self.model_size} 切换到 {opt_model_size}")
                    self.model_size = opt_model_size
                    model_classes = {
                        "rfdetr-nano": RFDETRNano,
                        "rfdetr-small": RFDETRSmall,
                        "rfdetr-medium": RFDETRMedium,
                        "rfdetr-large": RFDETRLarge,
                    }
                    self.model = model_classes[self.model_size]()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[*] 自动检测训练设备: {device}")
            
        # 根据 resize_mode 配置缩放与填充策略
        if resize_mode == "letterbox":
            square_resize_div_64 = False
            scale_jitter = False
            if aug_config is None:
                aug_config = {
                    "LongestMaxSize": {"max_size": img_size},
                    "PadIfNeeded": {
                        "min_height": img_size,
                        "min_width": img_size,
                        "border_mode": 0,
                        "value": [114, 114, 114],
                    },
                    "HorizontalFlip": {"p": 0.5},
                }
            print(f"[*] 启用 Letterbox 缩放模式 (保持长宽比缩放至最长边 {img_size}px + 灰色[114]填充至 {img_size}x{img_size})")
        elif resize_mode == "direct_resize":
            square_resize_div_64 = True
            scale_jitter = False
            print(f"[*] 启用直接拉伸缩放模式 (强制双线性插值直接缩放至 {img_size}x{img_size})")
        else:
            square_resize_div_64 = False
            print(f"[*] 使用自定义缩放配置 (resize_mode={resize_mode})")

        print(f"[*] 开始使用 {self.model_size} 训练，分辨率: {img_size}x{img_size}，数据集: {self.dataset_dir}")
        print(f"[*] 训练超参数: epochs={epochs}, batch_size={batch_size}, grad_accum_steps={grad_accum_steps}, "
              f"effective_batch_size={batch_size * grad_accum_steps}, lr_drop={lr_drop}, warmup_epochs={warmup_epochs}")

        self.model.train(
            dataset_file="yolo",
            dataset_dir=self.dataset_dir,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,  # 梯度累加，稳定二分图匹配
            lr_drop=lr_drop,                    # 动态学习率衰减点
            warmup_epochs=warmup_epochs,        # 预热轮数
            output_dir=output_dir,
            progress_bar="tqdm",
            device=device,
            resolution=img_size,                # 800 分辨率
            num_workers=num_workers,
            early_stopping=early_stopping,      # 早停策略
            eval_interval=eval_interval,        # 验证间隔
            multi_scale=multi_scale,            # 单尺度/多尺度配置
            scale_jitter=scale_jitter,          # 尺度抖动
            square_resize_div_64=square_resize_div_64,
            aug_config=aug_config,              # Letterbox 增强管道
        )
        print(f"[*] 训练结束，模型保存至: {output_dir}")


if __name__ == "__main__":
    # 默认指向当前项目中的 A701_AVI_wpoint_detect 目录
    current_dir = Path(__file__).parent.resolve()
    dataset_path = str(current_dir / "dataset" / "A701_AVI_wpoint_detect")
    
    trainer = YoloHBBTrainer(
        model_size="rfdetr-medium",
        dataset_dir=dataset_path
    )
    
    # 启动训练：rfdetr-medium, 800x800, Letterbox 模式
    trainer.train(
        img_size=800,
        resize_mode="letterbox",
        auto_optimize=True,
        early_stopping=False,
        multi_scale=False,
    )
