import pandas as pd
import numpy as np
import os

def add_new_column_to_xlsx(file_path="../data/LikesEvents.xlsx", new_column_name=None):
    """
    功能：在xlsx文件中添加新列，并在新列第一行写入指定内容
    参数：
        file_path: xlsx文件路径
        new_column_name: 要添加的新列标题内容
    """
    try:
        # 读取整个Excel文件
        df = pd.read_excel(file_path)
        
        # 检查标题是否已存在
        if new_column_name in df.columns:
            print(f"警告：'{new_column_name}'列已存在，跳过创建")
            return
        
        # 添加新列，默认值为NaN（空）
        df[new_column_name] = np.nan
        
        # 保存回文件
        df.to_excel(file_path, index=False)
        
    except Exception as e:
        print(f"添加列时出错: {str(e)}")

def add_or_update_id(file_path="../data/LikesEvents.xlsx", id_value=None, update_col_name=None):
    """
    功能：检查ID是否存在，不存在则添加新行并设置值，存在则更新对应位置
    新增：处理"总数"列 - 新ID设为0，已存在ID加1
    
    参数：
        file_path: xlsx文件路径
        "ID": ID列名称 (如'ID')
        id_value: 要查找/添加的ID值
        update_col_name: 要设置值的列名称
        "总数": 总数列名称，默认为"总数"
    """
    try:
        # 读取整个Excel文件
        df = pd.read_excel(file_path)
        
        # 查找ID是否存在
        id_exists = id_value in df["ID"].values
        
        if id_exists:
            # 更新现有行
            row_index = df.index[df["ID"] == id_value].tolist()[0]
            
            # 更新总数列：原有值加1
            current_total = df.at[row_index, "总数"]
            if pd.isna(current_total):
                current_total = 0
            df.at[row_index, "总数"] = current_total + 1
            
            # 更新指定列
            df.at[row_index, update_col_name] = 1
            
            status = "updated"
        else:
            # 创建新行
            new_row = {col: None for col in df.columns}  # 初始化为空值
            
            # 设置ID和指定列的值
            new_row["ID"] = id_value
            new_row[update_col_name] = 1
            new_row["总数"] = 0  # 新ID总数设为0
            
            # 添加到DataFrame
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            status = "added"
        
        # 保存回文件
        df.to_excel(file_path, index=False)
        
        # 返回ID和操作结果
        return None
        
    except Exception as e:
        print(f"添加/更新ID时出错: {str(e)}")
        return None

# 示例用法
if __name__ == "__main__":

    add_new_column_to_xlsx(new_column_name="2025-06-26点赞")
    
    result = add_or_update_id(
        id_value="u10025",
        update_col_name="2025-06-26点赞",
    )