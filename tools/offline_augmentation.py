import os
import cv2
import glob
import tqdm
import argparse
import albumentations as A
import numpy as np

def load_yolo_labels(label_path):
    """读取 YOLO 格式标签 (class_id, x_center, y_center, width, height)"""
    bboxes = []
    class_ids = []
    if not os.path.exists(label_path):
        return bboxes, class_ids
        
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                # 修复标注可能稍微越界的问题 (或者精度问题产生负数)
                xmin = max(0.0, xc - w / 2)
                ymin = max(0.0, yc - h / 2)
                xmax = min(1.0, xc + w / 2)
                ymax = min(1.0, yc + h / 2)
                
                new_w = xmax - xmin
                new_h = ymax - ymin
                new_xc = xmin + new_w / 2
                new_yc = ymin + new_h / 2
                
                # 过滤掉面积无效的框
                if new_w > 1e-5 and new_h > 1e-5:
                    class_ids.append(int(parts[0]))
                    bboxes.append([new_xc, new_yc, new_w, new_h])
    return bboxes, class_ids

def save_yolo_labels(label_path, bboxes, class_ids):
    """保存 YOLO 格式标签"""
    with open(label_path, "w", encoding="utf-8") as f:
        for bbox, class_id in zip(bboxes, class_ids):
            # 格式: class_id x_center y_center width height
            line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
            f.write(line)

def main():
    parser = argparse.ArgumentParser(description="离线数据增强 (支持 YOLO 标签)")
    parser.add_argument("--image_dir", type=str, default=r"dataset\A701_AVI_wpoint_detect\images\train", help="原始图片目录")
    parser.add_argument("--label_dir", type=str, default=r"dataset\A701_AVI_wpoint_detect\labels\train", help="原始标签目录")
    parser.add_argument("--output_image_dir", type=str, default=r"dataset\A701_AVI_wpoint_detect_aug\images\train", help="增强后图片保存目录")
    parser.add_argument("--output_label_dir", type=str, default=r"dataset\A701_AVI_wpoint_detect_aug\labels\train", help="增强后标签保存目录")
    parser.add_argument("--aug_times", type=int, default=4, help="每张图片增强生成的倍数")
    args = parser.parse_args()

    # 1. 检查并创建输出目录
    os.makedirs(args.output_image_dir, exist_ok=True)
    os.makedirs(args.output_label_dir, exist_ok=True)
    
    # 获取所有图片路径
    image_paths = glob.glob(os.path.join(args.image_dir, "*.*"))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if len(image_paths) == 0:
        print(f"在 {args.image_dir} 下未找到图片，请检查路径。")
        return

    # 2. 定义 Albumentations 增强流水线
    # 根据讨论，加入：亮暗对比度变化、高斯模糊/噪声、平移缩放旋转
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(p=1.0) # 去掉旧版本参数，使用默认方差
        ], p=0.5),
        # 替换被废弃的 ShiftScaleRotate，改用 Affine
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), cval=0, p=0.7)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3, clip=True))

    print(f"找到 {len(image_paths)} 张原图，准备按 {args.aug_times} 倍进行离线增强...")
    
    # 3. 遍历图片进行增强
    for img_path in tqdm.tqdm(image_paths, desc="数据增强进度"):
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        label_path = os.path.join(args.label_dir, f"{name}.txt")
        
        # 使用支持中文路径的方式读取图片
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"\n警告：无法读取图像 {img_path}，跳过。")
            continue
            
        bboxes, class_ids = load_yolo_labels(label_path)
        
        # 3.1 原图也直接拷贝一份到输出目录，作为基础训练数据
        out_orig_img = os.path.join(args.output_image_dir, filename)
        out_orig_lbl = os.path.join(args.output_label_dir, f"{name}.txt")
        cv2.imencode(ext, img)[1].tofile(out_orig_img)
        save_yolo_labels(out_orig_lbl, bboxes, class_ids)
        
        # 如果没有标签，也允许增强，只是不传 bbox (如果是背景图)
        if len(bboxes) == 0:
            for i in range(args.aug_times):
                transformed = transform(image=img, bboxes=[], class_labels=[])
                aug_img = transformed['image']
                
                out_img_path = os.path.join(args.output_image_dir, f"{name}_aug_{i}{ext}")
                out_lbl_path = os.path.join(args.output_label_dir, f"{name}_aug_{i}.txt")
                
                cv2.imencode(ext, aug_img)[1].tofile(out_img_path)
                save_yolo_labels(out_lbl_path, [], [])
            continue

        # 3.2 进行 N 次数据增强
        for i in range(args.aug_times):
            try:
                transformed = transform(image=img, bboxes=bboxes, class_labels=class_ids)
                aug_img = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_classes = transformed['class_labels']
                
                out_img_path = os.path.join(args.output_image_dir, f"{name}_aug_{i}{ext}")
                out_lbl_path = os.path.join(args.output_label_dir, f"{name}_aug_{i}.txt")
                
                cv2.imencode(ext, aug_img)[1].tofile(out_img_path)
                save_yolo_labels(out_lbl_path, aug_bboxes, aug_classes)
            except Exception as e:
                # 某些极端形变可能导致bbox非法，通常由albumentations内部抛出
                print(f"\n警告：图片 {filename} 在第 {i} 次增强时失败: {e}")
                
    print("\n离线数据增强完成！")
    print(f"增强后的图片已保存至: {args.output_image_dir}")
    print(f"增强后的标签已保存至: {args.output_label_dir}")

if __name__ == "__main__":
    main()
