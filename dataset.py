import os
import json
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image


class MyDataSetTrain(Dataset):
    def __init__(self, size, path):
        super().__init__()
        # 使用 os.path.join 确保路径兼容 Windows
        gt_path = os.path.join(path, "training", "cloth")
        sketch_path = os.path.join(path, "training", "sketch")
        texture_path = os.path.join(path, "training", "texture")

        # 【关键修改1】必须使用 sorted() 确保读取顺序一致
        # 这里假设三个文件夹内的文件名是完全对应的
        gt_filenames = sorted(os.listdir(gt_path))

        # 为了安全，也可以检查另外两个文件夹是否存在对应文件（可选，这里为了效率简化处理）

        self.gt_images = [os.path.join(gt_path, img) for img in gt_filenames]
        self.sketch_images = [os.path.join(sketch_path, img) for img in gt_filenames]
        self.texture_images = [os.path.join(texture_path, img) for img in gt_filenames]

        # 简单的长度检查
        # 注意：这里实际上检查的是生成的列表长度，因为是基于 gt_filenames 生成的，所以肯定相等。
        # 真正的检查是在 __getitem__ 里打开图片时
        assert len(self.gt_images) == len(self.sketch_images) == len(self.texture_images)

    def __len__(self):
        return len(self.gt_images)

    def check_pair(self, a, b, c):
        # 【关键修改2】使用 os.path.basename 处理文件名，兼容 Windows 和 Linux
        # 原始代码用 split('/') 在 Windows (用 \) 下会失效
        name_a = os.path.splitext(os.path.basename(a))[0]
        name_b = os.path.splitext(os.path.basename(b))[0]
        name_c = os.path.splitext(os.path.basename(c))[0]

        if not (name_a == name_b == name_c):
            print(f"Error: 文件名不匹配!")
            print(f"GT: {a}")
            print(f"Sketch: {b}")
            print(f"Texture: {c}")

        assert name_a == name_b == name_c

    def __getitem__(self, i):
        # 检查文件名是否对齐
        self.check_pair(self.gt_images[i], self.sketch_images[i], self.texture_images[i])

        # 读取并预处理图片
        gt_image = Image.open(self.gt_images[i]).convert("RGB")  # 确保转为RGB，防止RGBA报错
        gt_image = np.array(gt_image).astype(np.uint8)
        gt_image = (gt_image / 127.5 - 1.0).astype(np.float32)

        sketch_image = Image.open(self.sketch_images[i]).convert("L")  # 线稿通常是灰度图，或者保持RGB
        sketch_image = np.array(sketch_image).astype(np.uint8)
        # 如果线稿是单通道，可能需要 unsqueeze，视模型输入要求而定，这里保持原样
        sketch_image = (sketch_image / 127.5 - 1.0).astype(np.float32)

        texture_image = Image.open(self.texture_images[i]).convert("RGB")
        texture_image = np.array(texture_image).astype(np.uint8)
        texture_image = (texture_image / 127.5 - 1.0).astype(np.float32)

        return {
            "gt": gt_image,
            "sketch": sketch_image,
            "texture": texture_image
        }


class MyDataSetTrain(Dataset):
    def __init__(self, size, path, metadata_path=None):
        super().__init__()
        if metadata_path is None:
            metadata_path = os.path.join(path, "training", "train_bf_texture.json")

        with open(metadata_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        base_path = os.path.join(path, "training")
        self.gt_images = [os.path.join(base_path, item["cloth"]) for item in items]
        self.sketch_images = [os.path.join(base_path, item["sketch"]) for item in items]
        self.texture_images = [os.path.join(base_path, item["texture"]) for item in items]

    def __len__(self):
        return len(self.gt_images)

    def __getitem__(self, i):
        gt_image = Image.open(self.gt_images[i]).convert("RGB")
        gt_image = np.array(gt_image).astype(np.uint8)
        gt_image = (gt_image / 127.5 - 1.0).astype(np.float32)

        sketch_image = Image.open(self.sketch_images[i]).convert("L")
        sketch_image = np.array(sketch_image).astype(np.uint8)
        sketch_image = (sketch_image / 127.5 - 1.0).astype(np.float32)

        texture_image = Image.open(self.texture_images[i]).convert("RGB")
        texture_image = np.array(texture_image).astype(np.uint8)
        texture_image = (texture_image / 127.5 - 1.0).astype(np.float32)

        return {
            "gt": gt_image,
            "sketch": sketch_image,
            "texture": texture_image
        }


class MyDataSetValidation(Dataset):
    def __init__(self, size, path):
        super().__init__()
        gt_path = os.path.join(path, "validation", "gt")
        sketch_path = os.path.join(path, "validation", "sketch")
        texture_path = os.path.join(path, "validation", "texture")

        # 【关键修改1】同样加上 sorted
        gt_filenames = sorted(os.listdir(gt_path))

        self.gt_images = [os.path.join(gt_path, img) for img in gt_filenames]
        self.sketch_images = [os.path.join(sketch_path, img) for img in gt_filenames]
        self.texture_images = [os.path.join(texture_path, img) for img in gt_filenames]

        assert len(self.gt_images) == len(self.sketch_images) == len(self.texture_images)

    def __len__(self):
        return len(self.gt_images)

    def check_pair(self, a, b, c):
        # 【关键修改2】修复 Windows 路径解析问题
        name_a = os.path.splitext(os.path.basename(a))[0]
        name_b = os.path.splitext(os.path.basename(b))[0]
        name_c = os.path.splitext(os.path.basename(c))[0]

        if not (name_a == name_b == name_c):
            print(f"Error: Validation set mismatch!")
            print(f"GT: {a}")
            print(f"Sketch: {b}")
            print(f"Texture: {c}")

        assert name_a == name_b == name_c

    def __getitem__(self, i):
        self.check_pair(self.gt_images[i], self.sketch_images[i], self.texture_images[i])

        gt_image = Image.open(self.gt_images[i]).convert("RGB")
        gt_image = np.array(gt_image).astype(np.uint8)
        gt_image = (gt_image / 127.5 - 1.0).astype(np.float32)

        sketch_image = Image.open(self.sketch_images[i]).convert("L")
        sketch_image = np.array(sketch_image).astype(np.uint8)
        sketch_image = (sketch_image / 127.5 - 1.0).astype(np.float32)

        texture_image = Image.open(self.texture_images[i]).convert("RGB")
        texture_image = np.array(texture_image).astype(np.uint8)
        texture_image = (texture_image / 127.5 - 1.0).astype(np.float32)

        return {
            "gt": gt_image,
            "sketch": sketch_image,
            "texture": texture_image
        }
