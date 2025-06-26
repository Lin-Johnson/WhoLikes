from collections import Counter
from scipy import stats
import cv2
import numpy as np
import os

TARGET_SIZE = (128, 128)

def get_background_color_edge(img, edge_width=2):
    """获取图像边缘的背景色（适合纯色背景）"""
    # 获取图像边缘的像素
    top = img[:edge_width, :]
    bottom = img[-edge_width:, :]
    left = img[:, :edge_width]
    right = img[:, -edge_width:]
    
    # 合并所有边缘像素
    edge_pixels = np.vstack([top.reshape(-1, top.shape[-1]), 
                           bottom.reshape(-1, bottom.shape[-1]),
                           left.reshape(-1, left.shape[-1]),
                           right.reshape(-1, right.shape[-1])])
    
    # 使用众数作为背景色
    return stats.mode(edge_pixels, axis=0)[0][0]

def remove_background(img, bg_color, tolerance=8):
    """去除背景色（创建掩模）"""
    # 计算与背景色的差异
    diff = np.sum(np.abs(img - bg_color), axis=-1)
    # 创建背景掩模（True表示背景区域）
    mask = diff <= tolerance
    # 反转掩模（True表示前景区域）
    return ~mask

def merge_close_values(x, merge_num=10):
    if not x:
        return []
    
    x_sorted = sorted(x)
    result = []
    current_group = [x_sorted[0]]
    
    for num in x_sorted[1:]:
        if num - current_group[-1] < merge_num:
            current_group.append(num)
        else:
            # 计算当前组的众数
            if len(current_group) == 1:
                mode_val = current_group[0]
            else:
                counter = Counter(current_group)
                mode_val = counter.most_common(1)[0][0]
            result.append(mode_val)
            current_group = [num]
    
    # 处理最后一组
    if current_group:
        if len(current_group) == 1:
            mode_val = current_group[0]
        else:
            counter = Counter(current_group)
            mode_val = counter.most_common(1)[0][0]
        result.append(mode_val)

    # 计算相邻差值
    diffs = [result[i+1] - result[i] for i in range(len(result)-1)]

    # 计算众数
    diff_counts = Counter(diffs)
    most_common_diff = diff_counts.most_common(1)[0][0]  # 众数
    count = diff_counts.most_common(1)[0][1]             # 出现次数

    print("相邻差值:", diffs)
    print("众数:", most_common_diff)
    print("出现次数:", count)
    
    return result, most_common_diff, count

def split_subimages(img_path, output_dir, min_area=8000, debug=True):
    """
    切割大图中的不规则分布小图
    :param img_path: 大图路径
    :param output_dir: 小图保存目录
    :param min_area: 最小有效区域面积（过滤噪声）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(img_path)
    # 读取图像
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    print(f"输入图像宽度: {width}, 高度: {height}")
    
    # 获取背景色
    bg_color = get_background_color_edge(img)
    print(f"背景色为: {bg_color}")
    
    if bg_color < 300:
        # 去除背景（创建前景掩模）
        foreground_mask = remove_background(img, bg_color)
        thresh = foreground_mask.astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 二值化处理（自适应阈值应对光照变化）
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 3
        )
        
    # 形态学操作（可选）
    kernel = np.ones((1,1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓（只在前景区域）
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 过滤轮廓并收集尺寸数据
    valid_contours = []
    dimensions = []
    _x = []
    _y = []
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        if area > min_area:
            valid_contours.append(cnt)
            dimensions.append(w)
            dimensions.append(h)
            _x.append(x)
            _y.append(y)
            # print(f"轮廓 {i}: 位置({x}, {y}), 大小({w}x{h}), 面积={area}")

    # 计算尺寸的众数
    if dimensions:
        counts = Counter(dimensions)
        mode = counts.most_common(1)[0][0]
        print(f"\n所有轮廓尺寸的众数是: {mode}")
    else:
        print("\n没有找到有效轮廓")
        return

    # 合并相近坐标并扩展网格
    _x, x_diff, _ = merge_close_values(_x)
    _y, y_diff, _ = merge_close_values(_y)
    most_common_diff = x_diff if x_diff > y_diff else y_diff
    
    # 扩展网格边界
    while _x[0] - most_common_diff >= 0:
        _x.insert(0, _x[0] - most_common_diff)
    while _x[-1] + most_common_diff + mode < width:
        _x.append(_x[-1] + most_common_diff)
    while _y[0] - most_common_diff >= 0:
        _y.insert(0, _y[0] - most_common_diff)
    while _y[-1] + most_common_diff + mode < height:
        _y.append(_y[-1] + most_common_diff)

    # 生成所有切割位置
    cut_positions = [(x, y) for x in _x for y in _y]
    contour_points_sorted = sorted(cut_positions, key=lambda p: (p[1]//20, p[0]))

    # 保存切割结果
    final_contour_img = img.copy()
    sub_img_num = 0
    
    for i, (x, y) in enumerate(contour_points_sorted):
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + mode), min(height, y + mode)
        
        if x2 > x1 and y2 > y1:
            # 绘制切割区域
            cv2.rectangle(final_contour_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(final_contour_img, str(i), (x1+5, y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 检查是否为背景
            sub_img = img[y1:y2, x1:x2]
            if not np.allclose(sub_img, bg_color, atol=10):
                resized_img = cv2.resize(sub_img, TARGET_SIZE, 
                                    interpolation=cv2.INTER_AREA)
                cv2.imwrite(f"{output_dir}/sub_{i:03d}.jpg", resized_img)
                sub_img_num += 1

    print(f"切割完成！共保存 {sub_img_num} 张小图")
    
    if debug:
        # 创建调试窗口
        cv2.namedWindow("Processing", cv2.WINDOW_NORMAL)
        
        # 显示原始图像
        cv2.imshow("Processing", img)
        cv2.waitKey(0)
        
        # # 显示灰度图像
        # cv2.imshow("Processing", gray)
        # cv2.waitKey(0)
        
        # 显示二值化结果
        cv2.imshow("Processing", thresh)
        cv2.waitKey(0)
        
        # 显示形态学处理结果
        cv2.imshow("Processing", cleaned)
        cv2.waitKey(0)
        
        # 绘制轮廓
        contour_img = img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)
        cv2.imshow("Processing", contour_img)
        cv2.waitKey(0)
        
        # 绘制有效轮廓
        valid_contour_img = img.copy()
        cv2.drawContours(valid_contour_img, valid_contours, -1, (0, 0, 255), 3)
        cv2.imshow("Processing", valid_contour_img)
        cv2.waitKey(0)

        # 显示最终切割轮廓
        cv2.imshow("Processing", final_contour_img)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    split_subimages("image.jpg", "sub_images", min_area=8000, debug=True)