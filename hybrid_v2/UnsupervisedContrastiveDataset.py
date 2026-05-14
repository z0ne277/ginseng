"""
改进的无监督对比学习数据集

改进点：
1. 保留原有的增强操作
2. 优化形态学操作的频率
3. 添加拓扑保持的增强策略
"""

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageOps
import random
import numpy as np
import cv2


class UnsupervisedContrastiveDataset(Dataset):
    """
    改进的无监督对比学习数据集

    设计原则：
    1. 增强应该保持拓扑结构不变
    2. 形态学操作应该增强拓扑特征的学习
    3. 避免过度增强导致拓扑信息丢失
    """

    def __init__(
            self,
            csv_file,
            transform=None,
            use_augment=False,
            use_binarization=False,
            binarization_threshold=128,
            augment_strength='medium'  # 'light', 'medium', 'strong'
    ):
        """
        Args:
            csv_file: CSV文件路径
            transform: 图像变换pipeline
            use_augment: 是否使用数据增强
            use_binarization: 是否进行二值化预处理
            binarization_threshold: 二值化阈值 (0-255)
            augment_strength: 增强强度
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.use_augment = use_augment
        self.use_binarization = use_binarization
        self.binarization_threshold = binarization_threshold
        self.augment_strength = augment_strength

        # 根据增强强度设置参数
        self._set_augment_params()

    def _set_augment_params(self):
        """根据增强强度设置参数"""
        if self.augment_strength == 'light':
            self.flip_prob = 0.3
            self.rotate_prob = 0.3
            self.perspective_prob = 0.2
            self.noise_prob = 0.1
            self.morph_prob = 0.4
            self.max_rotation = 15
        elif self.augment_strength == 'medium':
            self.flip_prob = 0.5
            self.rotate_prob = 0.5
            self.perspective_prob = 0.4
            self.noise_prob = 0.2
            self.morph_prob = 0.6
            self.max_rotation = 30
        else:  # strong
            self.flip_prob = 0.6
            self.rotate_prob = 0.6
            self.perspective_prob = 0.5
            self.noise_prob = 0.3
            self.morph_prob = 0.7
            self.max_rotation = 45

    def __len__(self):
        return len(self.data)

    def _binarize(self, image):
        """自适应二值化"""
        img_np = np.array(image)
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        binary = cv2.adaptiveThreshold(
            img_np,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        return Image.fromarray(binary)

    def augment_image(self, image):
        """
        拓扑保持的数据增强策略
        """
        # 确保是灰度图
        if image.mode != 'L':
            image = image.convert('L')

        width, height = image.size

        # ---------- 0. 可选的预处理二值化 ----------
        if self.use_binarization and random.random() > 0.7:
            image = self._binarize(image)

        # ---------- 1. 基础几何变换（保持拓扑） ----------
        if random.random() < self.flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < self.flip_prob:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < self.rotate_prob:
            angle = random.randint(-self.max_rotation, self.max_rotation)
            image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=0)

        # ---------- 2. 透视变换（保留拓扑结构） ----------
        if random.random() < self.perspective_prob:
            img_np = np.array(image)
            distortion_scale = random.uniform(0.05, 0.15)
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
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            image = Image.fromarray(img_np)

        # ---------- 3. 轻微高斯噪声 ----------
        if random.random() < self.noise_prob:
            img_np = np.array(image).astype(np.float32)
            noise = np.random.normal(0, random.uniform(5, 15), img_np.shape)
            img_np = img_np + noise
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_np)

        # ---------- 4. 轻微模糊 ----------
        if random.random() < 0.3:
            radius = random.uniform(0.3, 1.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        # ---------- 5. 缩放（保持拓扑） ----------
        if random.random() < 0.4:
            scale_factor = random.uniform(0.9, 1.1)
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

        # ---------- 6. 小区域随机擦除（不影响整体拓扑） ----------
        if random.random() < 0.2:
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            for _ in range(random.randint(1, 2)):
                erase_size_h = random.randint(h // 30, h // 15)
                erase_size_w = random.randint(w // 30, w // 15)
                y = random.randint(0, max(0, h - erase_size_h))
                x = random.randint(0, max(0, w - erase_size_w))
                img_np[y:y + erase_size_h, x:x + erase_size_w] = 0
            image = Image.fromarray(img_np)

        # ---------- 7. 形态学操作（核心：拓扑特征增强） ----------
        if random.random() < self.morph_prob:
            img_np = np.array(image)
            kernel_size = random.choice([3, 5])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            op = random.choice(['open', 'close', 'erode', 'dilate', 'gradient'])
            if op == 'open':
                img_np = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)
            elif op == 'close':
                img_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)
            elif op == 'erode':
                img_np = cv2.erode(img_np, kernel, iterations=1)
            elif op == 'dilate':
                img_np = cv2.dilate(img_np, kernel, iterations=1)
            elif op == 'gradient':
                # 形态学梯度突出边缘
                img_np = cv2.morphologyEx(img_np, cv2.MORPH_GRADIENT, kernel)

            image = Image.fromarray(img_np)

        # ---------- 8. 对比度调整 ----------
        if random.random() < 0.3:
            image = ImageOps.autocontrast(image)

        # ---------- 9. 亮度微调 ----------
        if random.random() < 0.2:
            img_np = np.array(image).astype(np.float32)
            brightness_factor = random.uniform(0.9, 1.1)
            img_np = img_np * brightness_factor
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_np)

        # 保证输出是三通道
        image = image.convert("L").convert("RGB")
        return image

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image']
        image = Image.open(img_path).convert("L")

        if self.use_augment:
            # 生成两个增强视图
            img1 = self.augment_image(image.copy())
            img2 = self.augment_image(image.copy())
        else:
            img1 = image.convert("RGB")
            img2 = image.convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2


class SupervisedContrastiveDataset(Dataset):
    """
    有监督对比学习数据集（如果有标签）
    """

    def __init__(
            self,
            csv_file,
            transform=None,
            use_augment=True,
            label_column='label'
    ):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.use_augment = use_augment
        self.label_column = label_column

        # 如果有标签列
        if label_column in self.data.columns:
            self.labels = self.data[label_column].values
            self.has_labels = True
        else:
            self.labels = None
            self.has_labels = False

    def __len__(self):
        return len(self.data)

    def augment_image(self, image):
        """简化的增强"""
        if image.mode != 'L':
            image = image.convert('L')

        width, height = image.size

        # 基础变换
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            angle = random.randint(-20, 20)
            image = image.rotate(angle, resample=Image.BICUBIC, fillcolor=0)

        # 形态学操作
        if random.random() > 0.4:
            img_np = np.array(image)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            if random.random() > 0.5:
                img_np = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)
            else:
                img_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)
            image = Image.fromarray(img_np)

        return image.convert("RGB")

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image']
        image = Image.open(img_path).convert("L")

        if self.use_augment:
            img1 = self.augment_image(image.copy())
            img2 = self.augment_image(image.copy())
        else:
            img1 = image.convert("RGB")
            img2 = image.convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.has_labels:
            label = self.labels[idx]
            return img1, img2, label
        else:
            return img1, img2
