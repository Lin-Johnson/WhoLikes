from scripts.split import split_subimages
from scripts.image_deduplicator import process_images
from pathlib import Path

import os



if __name__ == '__main__':

    image_folder = "images"
    output_dir = "data\sub_images"

    # 1. 获取所有点赞头像
    for filename in os.listdir(image_folder):
        filepath = os.path.join(image_folder, filename)

        output_file_dir = os.path.join(output_dir, Path(filename).stem)
        save_file_dir = os.path.join(output_dir, "save")
        
        split_subimages(filepath, output_file_dir, min_area=5000, debug=False)
        
        process_images(output_file_dir, save_file_dir)

            



    