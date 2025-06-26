import os
import shutil
import numpy as np
import cv2
import string
import imagehash
import random
from PIL import Image
import concurrent.futures
from skimage.metrics import structural_similarity as ssim
from skimage import transform

def get_image_hash(image_path, hash_size=8):
    """计算多种感知哈希值组合"""
    try:
        with Image.open(image_path) as img:
            # 计算多种哈希值组合
            phash = str(imagehash.phash(img, hash_size=hash_size))
            ahash = str(imagehash.average_hash(img, hash_size=hash_size))
            dhash = str(imagehash.dhash(img, hash_size=hash_size))
            
            # 调整图像大小后进行特征检测
            orb_features = extract_orb_features(np.array(img), max_features=20)
            
            return {
                'phash': phash,
                'ahash': ahash,
                'dhash': dhash,
                'orb': orb_features
            }
    except Exception as e:
        print(f"无法处理图片 {image_path}: {str(e)}")
        return None

def extract_orb_features(image, max_features=20):
    """使用ORB算法提取关键点特征"""
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    # 使用ORB提取特征
    orb = cv2.ORB_create(max_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    # 返回描述子（如果没有特征则返回空）
    if descriptors is None:
        return np.array([])
    
    # 返回固定数量的特征点
    return descriptors[:max_features].flatten()

def resize_image(image, max_dim=512):
    """调整图像到合理大小，保持宽高比"""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    
    scale = max_dim / max(h, w)
    return transform.resize(image, (int(h * scale), int(w * scale)))

def compare_images(img1, img2, sift_ratio=0.6, min_matches=12):
    try:
        if img1 is None or img2 is None:
            return 0.0
            
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # SIFT特征检测
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        # 如果某张图片没有特征点
        if des1 is None or des2 is None:
            return 0.0
            
        # FLANN匹配器
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Lowe's比率测试
        good_matches = [m for m, n in matches if m.distance < sift_ratio * n.distance]
        
        if len(good_matches) < min_matches:
            return 0.0
            
        # 提取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        
        # 计算单应性矩阵并验证
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None or not isinstance(H, np.ndarray) or H.shape != (3, 3):
            return 0.0
            
        # 检查变换矩阵合理性
        if abs(np.linalg.det(H)) < 1e-6:  # 避免奇异矩阵
            return 0.0
            
        # 执行透视变换
        aligned_img = cv2.warpPerspective(img1, H.astype(np.float32), 
                                        (img2.shape[1], img2.shape[0]))
                                        
        # 计算SSIM
        aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(aligned_gray, gray2, full=True, data_range=255)
        return max(0.0, min(score, 1.0))
        
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        return 0.0

def is_similar_images(features1, features2):
    """
    多种方法组合判断图片相似性：
    1. 哈希值汉明距离
    2. 结构相似性
    
    返回: True/False 图片是否相似
    """
    # 方法1: 哈希值匹配
    thresholds = {'ahash': 3, 'dhash': 5, 'phash': 10}
    distance_sum = 0
    distance_threshold = 30
    for key in ['dhash', 'phash']:
        if key in features1 and key in features2:
            try:
                hash1 = features1[key]
                hash2 = features2[key]
                hash1_obj = imagehash.hex_to_hash(hash1)
                hash2_obj = imagehash.hex_to_hash(hash2)
                distance = hash1_obj - hash2_obj
                distance_sum += distance
                if distance < thresholds[key]:
                    return True
            except Exception as e:
                print(f"Error in {key}: {e}")
            
    # 方法2: ​​SIFT+SSIM​
    if 'image' in features1 and 'image' in features2:
        image1 = features1['image']
        image2 = features2['image']
        
        similarity = compare_images(image1, image2)
        if similarity > 0.8:  # SSIM阈值
            
            return True
    
    return False

def count_orb_matches(descriptors1, descriptors2):
    """计算ORB特征匹配数量"""
    # 重新组织描述子
    d1 = descriptors1.reshape(-1, 32).astype(np.uint8)
    d2 = descriptors2.reshape(-1, 32).astype(np.uint8)
    
    # 使用BFMatcher进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    
    return len(matches)

def hist_correlation(img1, img2):
    """计算直方图相关性（对光照变化鲁棒）"""
    # 转换为HSV颜色空间
    if len(img1.shape) == 3:
        img1_hsv = cv2.cvtColor(img1.astype(np.float32), cv2.COLOR_RGB2HSV)
        img2_hsv = cv2.cvtColor(img2.astype(np.float32), cv2.COLOR_RGB2HSV)
    else:
        img1_hsv = img1
        img2_hsv = img2
    
    # 计算直方图
    hist1 = cv2.calcHist([img1_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([img2_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    
    # 归一化直方图
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # 计算相关性
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def process_images(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输出目录中所有图片的特征
    output_features = []
    print("正在分析输出目录图片...")
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        if not os.path.isfile(filepath):
            continue
        
        # 载入图像像素数据（仅用于相似性检测）
        img = cv2.imread(filepath)
        if img is None:
            continue
            
        features = {
            'filepath': filepath,
            'features': get_image_hash(filepath)
        }
        features['features']['image'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_features.append(features)
    
    print(f"输出目录中已有 {len(output_features)} 张图片作为基准")
    
    # 使用线程池并行处理图片
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # 遍历输入目录中的图片
        futures = []
        for filename in os.listdir(input_dir):
            input_path = os.path.join(input_dir, filename)
            if not os.path.isfile(input_path):
                continue
            
            # 提交任务给线程池
            futures.append(executor.submit(
                process_single_image, 
                input_path, 
                output_dir, 
                output_features
            ))
        
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                print(result)

# 生成随机字符串的函数
def generate_random_string(length=8):
    """生成指定长度的随机字符串（包含大小写字母和数字）"""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

# 修改后的处理单张图片函数
def process_single_image(input_path, output_dir, output_features):
    """处理单个图片的复制逻辑 - 简化版"""
    # 检查是否有效图片
    img = cv2.imread(input_path)
    if img is None:
        return f"跳过非图片文件: {os.path.basename(input_path)}"
    
    # 提取特征
    input_features = get_image_hash(input_path)
    if input_features is None:
        return f"无法提取特征: {os.path.basename(input_path)}"
    
    # 添加图像数据用于相似性检测
    input_features['image'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    # 与所有输出图片比较
    for output in output_features:
        if is_similar_images(input_features, output['features']):
            # 获取相似图片的文件名
            original_filename = os.path.basename(output['filepath'])
            
            # 如果文件名符合新格式，则增加前缀
            if '_' in original_filename and original_filename.split('_')[0].isdigit():
                prefix = original_filename.split('_')[0]
                # 增加前缀值
                new_prefix = str(int(prefix) + 1).zfill(4)
                new_filename = f"{new_prefix}_{'_'.join(original_filename.split('_')[1:])}"
                
                # 构建完整路径
                new_filepath = os.path.join(output_dir, new_filename)
                old_filepath = os.path.join(output_dir, original_filename)
                
                # 重命名文件
                os.rename(old_filepath, new_filepath)
                
                # 更新特征记录中的文件路径
                output['filepath'] = new_filepath
                
                return (f"存在相似图片: {os.path.basename(input_path)} "
                        f"≈ {original_filename} -> 重命名为: {new_filename}")
            else:
                return (f"存在相似图片(但不符合命名规则): {os.path.basename(input_path)} "
                        f"≈ {original_filename}")
    
    # 如果没有发现相似图片，添加新图片
    # 创建新文件名的前缀部分
    new_prefix = 1
    # 创建完整文件名
    base_name, ext = os.path.splitext(os.path.basename(input_path))
    new_filename = f"{str(new_prefix).zfill(4)}_{generate_random_string()}{ext}"
    output_path = os.path.join(output_dir, new_filename)
    
    # 处理文件名冲突（小概率事件）
    while os.path.exists(output_path):
        new_filename = f"{str(new_prefix).zfill(4)}_{generate_random_string()}{ext}"
        output_path = os.path.join(output_dir, new_filename)
    
    # 复制图片到输出目录
    shutil.copy2(input_path, output_path)
    
    # 添加新特征到输出列表
    new_features = {
        'filepath': output_path,
        'features': input_features
    }
    output_features.append(new_features)
    
    return f"添加新图片: {os.path.basename(input_path)} -> {new_filename}"

if __name__ == "__main__":
    # input_directory = "../sub_images/240616"  # 输入图片目录
    # output_directory = "../sub_images/save"  # 输出图片目录

    input_directory = "../sub_images/test_input"  # 输入图片目录
    output_directory = "../sub_images/test_output"  # 输出图片目录
    
    process_images(input_directory, output_directory)