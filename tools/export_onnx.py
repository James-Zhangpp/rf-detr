import os
import torch
import torch.nn as nn
from rfdetr import RFDETR
import rfdetr.export.main

# === Monkey-patch 以支持导出时使用 NHWC 作为 dummy_input ===
original_make_infer_image = rfdetr.export.main.make_infer_image

def patched_make_infer_image(*args, **kwargs):
    # 原始生成的是 NCHW [B, C, H, W]
    tensor = original_make_infer_image(*args, **kwargs)
    # 转置为 NHWC [B, H, W, C]
    return tensor.permute(0, 2, 3, 1)

rfdetr.export.main.make_infer_image = patched_make_infer_image

class NormalizedModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        # ImageNet mean and std
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, *args, **kwargs):
        images = args[0]
        # === 核心修改：模型接受 NHWC (1xHxWxC) 输入，并在此转换为 NCHW (1xCxHxW) ===
        images = images.permute(0, 3, 1, 2)
        
        # 部署推理时，假设输入的是 [0, 255] 的像素值 (可以是 uint8 或 float)
        # 将其缩放到 [0, 1] 然后应用减均值除以方差的归一化
        images = images.float() / 255.0
        images = (images - self.mean) / self.std
        
        # 将处理后的图像放回参数并传给原模型
        new_args = (images,) + args[1:]
        return self.base_model(*new_args, **kwargs)
        
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

def main():
    # === 参数配置 ===
    project_name = "A701_AVI_wpoint_detect"  # 修改为您的项目/数据集名称
    img_size = 640                           # 修改为您训练时实际使用的图像大小 (如 640, 800, 1024)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 设置模型权重路径 (自动定位到项目的 best_total.pth)
    checkpoint_path = os.path.join(current_dir, "output", project_name, "checkpoint_best_total.pth")
    
    # 2. 设置输出目录 (按项目名称隔离)
    output_dir = os.path.join(current_dir, "export", project_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {checkpoint_path}...")
    # 使用 from_checkpoint 加载模型权重
    model = RFDETR.from_checkpoint(checkpoint_path)
    
    # === 新增逻辑：将归一化和格式转换(NHWC->NCHW)嵌入到 ONNX 内部 ===
    print("Embedding normalization and NHWC->NCHW conversion into the model graph...")
    model.model.model = NormalizedModel(model.model.model)
    
    print("Exporting model to ONNX format...")
    # 3. 导出模型 (按照我们讨论的参数设置：固定 batch size=1, 默认分辨率 shape=None)
    export_path = model.export(
        output_dir=output_dir,
        format="onnx",
        batch_size=1,
        dynamic_batch=False,
        shape=(img_size, img_size),  # 显式指定输入分辨率，而不是 None
        opset_version=17,
        verbose=True
    )
    
    print(f"Export successfully completed! ONNX model saved at: {export_path}")

if __name__ == "__main__":
    main()
