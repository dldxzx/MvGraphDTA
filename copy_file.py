import shutil
import os
import pandas as pd

# def copy_folder(src_folder, dest_folder):
#     """
#     复制指定名字文件夹（包括文件夹中的所有内容）到指定的地址。

#     :param src_folder: 源文件夹路径
#     :param dest_folder: 目标文件夹路径
#     """
#     try:
#         # 检查目标文件夹是否存在
#         if not os.path.exists(dest_folder):
#             # 复制整个文件夹到目标地址
#             shutil.copytree(src_folder, dest_folder)
#             print(f"文件夹已成功复制到 {dest_folder}")
#         else:
#             print(f"目标文件夹 {dest_folder} 已存在")
#     except Exception as e:
#         print(f"复制文件夹时发生错误: {e}")

import shutil
import os

def copy_file(src_file, dest_file):
    """
    复制指定名字的文件到指定的地址。

    :param src_file: 源文件路径
    :param dest_file: 目标文件路径
    """
    try:
        # 检查源文件是否存在
        if os.path.exists(src_file):
            # 复制文件到目标地址
            shutil.copy(src_file, dest_file)
            print(f"文件已成功复制到 {dest_file}")
        else:
            print(f"源文件 {src_file} 不存在")
    except Exception as e:
        print(f"复制文件时发生错误: {e}")

# /home/user/zky/rwrSage/data_1/drug_edge_feature
# 示例用法
source_folder = "/home/user/zky/rwrSage/data_2/target_node_edge_index/"
destination_folder = "/home/user/zky/MvGraphDTA/data/binding_affinity/li_data/casf2016/target_node_edge_index/"

test_ids = pd.read_csv('/home/user/zky/MvGraphDTA/data/binding_affinity/li_data/casf2016.csv')['PDBID'].to_numpy().tolist()
print(test_ids)

for id in test_ids:
    temp_source_folder = source_folder + id + '.pt'
    copy_file(temp_source_folder, destination_folder)
    # break
