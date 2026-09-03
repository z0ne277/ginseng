import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageOps
import random
import numpy as np
import cv2


class UnsupervisedContrastiveDataset(Dataset):
    """
    改进的无监督对比学习数据集

    改进点：
    1. 保留原有的增强操作
    2. 增强形态学操作的权重（腐蚀/膨胀更频繁）
    3. 添加自适应二值化选项
    4. 更好的纹理保留策略
    """

    def __init__(
            self,
            csv_file,
            transform=None,
            use_augment=False,
            use_binarization=False,
            binarization_threshold=128
    ):
        """
        Args:
            csv_file: CSV文件路径
            transform: 图像变换pipeline
            use_augment: 是否使用数据增强
            use_binarization: 是否进行二值化预处理
            binarization_threshold: 二值化阈值 (0-255)
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.use_augment = use_augment
        self.use_binarization = use_binarization
        self.binarization_threshold = binarization_threshold

    def __len__(self):
        return len(self.data)

    def _binarize(self, image):
        """
        自适应二值化：保留二值特性
        两种方式：
        1. 固定阈值（简单快速）
        2. 自适应阈值（更鲁棒）
        """
        img_np = np.array(image)

        # 自适应二值化：更适合有阴影/光照变化的图像
        binary = cv2.adaptiveThreshold(
            img_np,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,  # 必须是奇数
            C=2
        )
        return Image.fromarray(binary)

    def augment_image(self, image):
        """
        增强的数据增强策略，更适合人参二值图
        """
        width, height = image.size

        # ---------- 0. 可选的预处理二值化 ----------
        if self.use_binarization and random.random() > 0.7:
            image = self._binarize(image)

        # ---------- 1. 基础几何变换 ----------
        if random.random() > 0.4:  # 提高概率
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.4:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            image = image.rotate(angle, resample=Image.BICUBIC, expand=False)

        # ---------- 2. 透视变换（保留拓扑结构） ----------
        if random.random() > 0.6:
            img_np = np.array(image)
            distortion_scale = random.uniform(0.08, 0.25)
            half_width = width // 2
            half_height = height // 2

            topleft = (
                random.randint(0, int(distortion_scale * half_width)),
                random.randint(0, int(distortion_scale * half_height))
            )
            topright = (
                width - random.randint(0, int(distortion_scale * half_width)),
                random.randint(0, int(distortion_scale * half_height))
            )
            botright = (
                width - random.randint(0, int(distortion_scale * half_width)),
                height - random.randint(0, int(distortion_scale * half_height))
            )
            botleft = (
                random.randint(0, int(distortion_scale * half_width)),
                height - random.randint(0, int(distortion_scale * half_height))
            )

            from_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            to_points = np.float32([topleft, topright, botright, botleft])
            M = cv2.getPerspectiveTransform(from_points, to_points)
            img_np = cv2.warpPerspective(
                img_np, M, (width, height),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
            )
            image = Image.fromarray(img_np)

        # ---------- 3. 高斯噪声（保持二值特性） ----------
        if random.random() > 0.7:
            img_np = np.array(image).astype(np.float32)
            noise = np.random.normal(0, random.uniform(8, 25), img_np.shape)
            img_np = img_np + noise
            img_np = np.clip(img_np, 0, 255)
            # 二值化以保持特性
            img_np = (img_np > 128).astype(np.uint8) * 255
            image = Image.fromarray(img_np.astype(np.uint8))

        # ---------- 4. 模糊 ----------
        if random.random() > 0.7:
            radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        # ---------- 5. 缩放和填充 ----------
        if random.random() > 0.5:
            scale_factor = random.uniform(0.85, 1.15)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = image.resize(new_size, Image.BICUBIC)

            if scale_factor > 1:
                left = (new_size[0] - width) // 2
                top = (new_size[1] - height) // 2
                right = left + width
                bottom = top + height
                image = image.crop((left, top, right, bottom))
            elif scale_factor < 1:
                new_image = Image.new('L', (width, height), 0)
                paste_left = (width - new_size[0]) // 2
                paste_top = (height - new_size[1]) // 2
                new_image.paste(image, (paste_left, paste_top))
                image = new_image

        # ---------- 6. 随机擦除 ----------
        if random.random() > 0.75:
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            for _ in range(random.randint(1, 3)):
                erase_size_h = random.randint(h // 25, h // 10)
                erase_size_w = random.randint(w // 25, w // 10)
                y = random.randint(0, max(0, h - erase_size_h))
                x = random.randint(0, max(0, w - erase_size_w))
                img_np[y:y + erase_size_h, x:x + erase_size_w] = 0
            image = Image.fromarray(img_np)

        # ---------- 7. 随机补块 ----------
        if random.random() > 0.8:
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            num_patches = random.randint(1, 3)
            for _ in range(num_patches):
                patch_h = random.randint(h // 25, h // 8)
                patch_w = random.randint(w // 25, w // 8)
                y = random.randint(0, max(0, h - patch_h))
                x = random.randint(0, max(0, w - patch_w))
                fill_value = 0 if random.random() > 0.5 else 255
                img_np[y:y + patch_h, x:x + patch_w] = fill_value
            image = Image.fromarray(img_np)

        # ---------- 8. 【增强】形态学操作（开/闭）- 更高频率 ----------
        # 这是为了保留/增强拓扑特征
        if random.random() > 0.3:  # 提高到70%概率
            img_np = np.array(image)
            kernel_size = random.choice([3, 5, 7])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            if random.random() > 0.5:
                # 开操作：去除细小细节，保留主体
                img_np = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)
            else:
                # 闭操作：填充小孔，平滑边界
                img_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)

            image = Image.fromarray(img_np)

        # ---------- 9. 【新增】形态学腐蚀/膨胀 - 更高频率 ----------
        # 这对于提取拓扑特征很重要
        if random.random() > 0.35:  # 65%概率
            img_np = np.array(image)
            kernel_size = random.choice([3, 5])
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

            if random.random() > 0.5:
                # 腐蚀：消除细节，保留主体
                img_np = cv2.erode(img_np, kernel, iterations=1)
            else:
                # 膨胀：填充孔洞，连接分离的部分
                img_np = cv2.dilate(img_np, kernel, iterations=1)

            image = Image.fromarray(img_np)

        # ---------- 10. 【新增】对比度增强 ----------
        if random.random() > 0.6:
            image = ImageOps.autocontrast(image)

        # 保证输出是三通道
        image = image.convert("L").convert("RGB")
        return image

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image']
        image = Image.open(img_path).convert("L").convert("RGB")

        if self.use_augment:
            img1 = self.augment_image(image)
            img2 = self.augment_image(image)
        else:
            img1 = image
            img2 = image

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2
