import cv2
import numpy as np
import onnxruntime as ort
import os
import glob
import time
import argparse

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """保持长宽比缩放图片，并在边缘填充颜色"""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def process_single_image(session, input_name, expected_H, expected_W, image_path, output_dir, threshold=0.5, max_det=None):
    """
    处理单张图像，运行推理，并把结果画框保存
    """
    # 用 cv2.imdecode 代替 cv2.imread 以支持中文路径
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: 无法读取图像 {image_path}")
        return
    
    # 拷贝一份干净的原图，专门用来做无损裁剪，防止被后期的画框和红圈污染
    img_clean = img.copy()
    filename = os.path.basename(image_path)
    
    orig_H, orig_W = img.shape[:2]
    
    # BGR 转换为 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 使用 Letterbox 等比例缩放并填充，防止图像变形
    img_letterboxed, r, (dw, dh) = letterbox(img_rgb, new_shape=(expected_W, expected_H))
    # 判断模型需要的格式，如果是NCHW，则需要transpose
    if expected_W == img_letterboxed.shape[1] and expected_H == img_letterboxed.shape[0]:
        # 虽然这里判断不太严谨，但假设前面已正确获取 expected_H, expected_W
        # 根据ONNX输入形状决定是否转置
        pass
        
    input_tensor = img_letterboxed[np.newaxis, ...].astype(np.float32)
    # 动态检查session input shape
    in_shape = session.get_inputs()[0].shape
    if len(in_shape) == 4 and in_shape[1] == 3: # NCHW
        input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))
    
    # 执行推理
    start_time = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    infer_time = (time.time() - start_time) * 1000.0  # 转换为毫秒
    
    # 提取输出
    output_names = [out.name for out in session.get_outputs()]
    boxes_idx = next(i for i, name in enumerate(output_names) if "dets" in name)
    logits_idx = next(i for i, name in enumerate(output_names) if "labels" in name)
    
    boxes_cwh = outputs[boxes_idx][0]
    logits = outputs[logits_idx][0]
    
    # 如果最后一个维度是背景类（某些DETR实现），且类别数大于1，才可能需要:-1，
    # 但一般情况RT-DETR/RF-DETR导出的ONNX没有背景类，直接用全部logits。
    # 如果只有1个类别，切片:-1会导致维度变成0，什么都检测不到。
    
    # 后处理
    scores = 1 / (1 + np.exp(-logits))
    max_scores = scores.max(axis=-1)
    class_ids = scores.argmax(axis=-1)
    
    # 设定阈值过滤
    keep = max_scores > threshold
    boxes_cwh = boxes_cwh[keep]
    max_scores = max_scores[keep]
    class_ids = class_ids[keep]
    
    # --- 新增：Top-K 数量限制 (解决数量固定时的多抓问题) ---
    if max_det is not None and len(max_scores) > max_det:
        # 按置信度从高到低排序，截取前 max_det 个
        sorted_indices = np.argsort(max_scores)[::-1]
        top_k = sorted_indices[:max_det]
        
        boxes_cwh = boxes_cwh[top_k]
        max_scores = max_scores[top_k]
        class_ids = class_ids[top_k]
    # -----------------------------------------------------------
    
    # 画框并保存
    if len(boxes_cwh) > 0:
        cx, cy, bw, bh = boxes_cwh.T
        
        # 1. 还原坐标：ONNX 输出通常是基于 letterbox 后图像尺寸的 0~1 比例
        # 先算出在 Letterbox 图像上的绝对坐标
        cx_abs = cx * expected_W
        cy_abs = cy * expected_H
        bw_abs = bw * expected_W
        bh_abs = bh * expected_H
        
        # 2. 去除黑边 (padding) 并除以缩放比例 (r) 映射回原始图像的像素坐标
        cx_orig = (cx_abs - dw) / r
        cy_orig = (cy_abs - dh) / r
        bw_orig = bw_abs / r
        bh_orig = bh_abs / r
        
        x1 = cx_orig - bw_orig / 2
        y1 = cy_orig - bh_orig / 2
        x2 = cx_orig + bw_orig / 2
        y2 = cy_orig + bh_orig / 2
        
        for i in range(len(max_scores)):
            # 限制坐标在图像边界范围内，防止越界
            px1 = max(0, int(x1[i]))
            py1 = max(0, int(y1[i]))
            px2 = min(orig_W, int(x2[i]))
            py2 = min(orig_H, int(y2[i]))
            
            box_area = (px2 - px1) * (py2 - py1)
            pt1 = (px1, py1)
            pt2 = (px2, py2)
            score = max_scores[i]
            cls_id = class_ids[i]
            
            # --- 新增：保存干干净净的 ROI 裁剪小图 ---
            if px2 > px1 and py2 > py1:
                # 建立类似 export_results/crops/图片名/ 的文件夹
                crop_dir = os.path.join(output_dir, "crops", os.path.splitext(filename)[0])
                os.makedirs(crop_dir, exist_ok=True)
                
                crop_img = img_clean[py1:py2, px1:px2]
                
                # 使用高质量三次插值 (Cubic) 将微小的焊点统一拉伸到 128x128
                # crop_img = cv2.resize(crop_img, (128, 128), interpolation=cv2.INTER_CUBIC)
                
                # 改用无损的 .png 格式保存，拒绝 JPEG 压缩带来的马赛克边缘
                crop_path = os.path.join(crop_dir, f"crop_{i:02d}.png")
                # cv2.imwrite(crop_path, crop_img) # 替换为支持中文路径的写法
                is_success, im_buf = cv2.imencode(".png", crop_img)
                if is_success:
                    im_buf.tofile(crop_path)
            # ------------------------------------------
            
            # --- 新增：提取 ROI，阈值分割，查找轮廓，计算中心点与面积 ---
            area_val = 0
            center_x, center_y = -1, -1
            if px2 > px1 and py2 > py1:
                roi = img[py1:py2, px1:px2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # 阈值设定为 40~70 算作灰色
                _, binary_roi = cv2.threshold(gray_roi, 60, 70, cv2.THRESH_BINARY)
                
                # 查找轮廓 (只查找外层轮廓)
                contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                    # 获取面积最大的轮廓
                    max_contour = max(contours, key=cv2.contourArea)
                    area_val = cv2.contourArea(max_contour)
                    
                    # 为了过滤掉极小的噪点，设定一个极小的面积下限
                    if area_val > 2:
                        # 使用图像矩(Moments)计算重心坐标
                        M = cv2.moments(max_contour)
                        if M["m00"] != 0:
                            # 局部 ROI 坐标
                            roi_cx = int(M["m10"] / M["m00"])
                            roi_cy = int(M["m01"] / M["m00"])
                            
                            # 映射回原图绝对坐标
                            center_x = px1 + roi_cx
                            center_y = py1 + roi_cy
                            
                            # 给轮廓所有的点加上 (px1, py1) 的偏移量，以便在原图上绘制
                            max_contour_offset = max_contour + np.array([[px1, py1]])
                            
                            # 用红色描边画出这个最大反光点的轮廓
                            cv2.drawContours(img, [max_contour_offset], -1, (0, 0, 255), 1)
                            # 用黄色实心圆标记重心位置
                            cv2.circle(img, (center_x, center_y), 2, (0, 255, 255), -1)
            # -----------------------------------------------------------
            
            # 画目标检测矩形框 (绿色)
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            
            # 标签：类别、分数、最大白点面积，以及计算出的质心坐标
            label = f"WArea:{area_val:.0f}"
            if center_x != -1:
                label += f" ({center_x},{center_y})"
                
            cv2.putText(img, label, (pt1[0], max(pt1[1] - 5, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        
    # 保存画了框的结果图，使用 imencode 支持中文路径
    save_path = os.path.join(output_dir, f"out_{filename}")
    _, ext = os.path.splitext(save_path)
    if not ext: ext = '.png'
    is_success, im_buf = cv2.imencode(ext, img)
    if is_success:
        im_buf.tofile(save_path)
    print(f"Processed {filename} -> {save_path} (Detected: {len(max_scores)}, Infer Time: {infer_time:.2f}ms)")

def main():
    parser = argparse.ArgumentParser(description="ONNX Inference")
    parser.add_argument("--source", type=str, default=r"C:\Users\ASUS\Desktop\A07_01\原图", help="手动输入测试图片或文件夹路径 (例如: C:/images/test.jpg)")
    args = parser.parse_args()

    # ================= 配置区 =================
    project_name = "A701_AVI_wpoint_detect"  # 修改为您的项目/数据集名称
    model_size = "rfdetr-small"              # 您导出的模型大小 (nano, small, medium)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ONNX 模型路径 (自动去 export/项目名/ 下找)
    onnx_path = os.path.join(current_dir, "export", project_name, f"{model_size}.onnx")
    
    # source_path 可以是【单张图片】的路径，也可以是【包含多张图片的文件夹】路径
    if args.source:
        source_path = args.source
    else:
        # 默认去找 dataset/项目名/images/train，如果不存在就退回到 dataset/项目名
        source_path = os.path.join(current_dir, "dataset", project_name, "images", "train")
        if not os.path.exists(source_path):
            source_path = os.path.join(current_dir, "dataset", project_name)
    
    # 结果图片保存目录 (按项目名称隔离)
    output_dir = os.path.join(current_dir, "export_results", project_name)
    
    # 推理阈值 (建议设低一点防止漏抓，比如 0.2 或 0.3)
    confidence_threshold = 0.3
    # 期望检测的最大目标数 (Top-K)。如果您确切知道盘上有 40 个焊点，设为 40 即可。
    expected_max_det = 40
    # =========================================
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(onnx_path):
        print(f"Error: 找不到模型文件 {onnx_path}")
        return
        
    print(f"Loading ONNX model from {onnx_path}...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    if input_shape[3] == 3:  # NHWC 格式 (B, H, W, C)
        _, expected_H, expected_W, _ = input_shape
    else:                    # NCHW 格式 (B, C, H, W)
        _, _, expected_H, expected_W = input_shape
    print(f"Model expects input shape: {expected_H}x{expected_W}")
    
    # 获取需要处理的图片路径列表
    image_paths = []
    if os.path.isfile(source_path):
        image_paths.append(source_path)
    elif os.path.isdir(source_path):
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp"]:
            image_paths.extend(glob.glob(os.path.join(source_path, ext)))
            image_paths.extend(glob.glob(os.path.join(source_path, ext.upper())))
    else:
        print(f"Error: 无效的输入路径 {source_path}")
        return
        
    if not image_paths:
        print(f"Warning: 在 {source_path} 中没有找到任何图片。")
        return
        
    print(f"Found {len(image_paths)} images, starting inference...\n")
    
    # 遍历处理所有图片
    for img_path in image_paths:
        process_single_image(
            session=session,
            input_name=input_name, 
            expected_H=expected_H, 
            expected_W=expected_W, 
            image_path=img_path, 
            output_dir=output_dir,
            threshold=confidence_threshold,
            max_det=expected_max_det
        )
        
    print(f"\nAll completed! Results are saved in {output_dir}")

if __name__ == "__main__":
    main()
