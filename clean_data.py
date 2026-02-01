import os

# 你的数据集根目录
dataset_root = r"D:\fuxian\datasets\BF"
# 需要检查的子数据集
sub_datasets = ["training", "validation"]
# 需要对齐的三个模态文件夹
modalities = ["gt", "sketch", "texture"]


def clean_folder(sub_name):
    base_path = os.path.join(dataset_root, sub_name)
    if not os.path.exists(base_path):
        print(f"跳过: 找不到文件夹 {base_path}")
        return

    print(f"正在检查 {sub_name} 数据集...")

    # 1. 读取所有文件名
    files_sets = []
    for mod in modalities:
        mod_path = os.path.join(base_path, mod)
        if not os.path.exists(mod_path):
            print(f"  错误: 缺少必要文件夹 {mod_path}")
            return

        # 获取所有文件名（不包含路径）
        files = set(os.listdir(mod_path))
        files_sets.append(files)
        print(f"  - {mod}: 发现 {len(files)} 张图片")

    # 2. 取交集（找出三个文件夹都有的文件）
    common_files = files_sets[0].intersection(files_sets[1]).intersection(files_sets[2])
    print(f"  => 有效匹配图片数: {len(common_files)}")

    if len(common_files) == 0:
        print("  警告: 没有找到任何匹配的图片！请检查文件名是否一致。")
        return

    # 3. 删除不匹配的文件
    deleted_count = 0
    for i, mod in enumerate(modalities):
        mod_path = os.path.join(base_path, mod)
        current_files = files_sets[i]

        # 找出不在交集中的文件
        files_to_remove = current_files - common_files

        for f in files_to_remove:
            file_path = os.path.join(mod_path, f)
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"  删除失败: {file_path}, 原因: {e}")

    if deleted_count > 0:
        print(f"  => 已清理 {deleted_count} 个不匹配的文件。")
    else:
        print("  => 文件完美匹配，无需清理。")
    print("-" * 30)


# 执行清理
for sub in sub_datasets:
    clean_folder(sub)

print("所有清理工作完成！")