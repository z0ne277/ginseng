import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import random
import numpy as np
import cv2

class UnsupervisedContrastiveDataset(Dataset):
    def __init__(self, csv_file, transform=None, use_augment=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.use_augment = use_augment

    def __len__(self):
        return len(self.data)

    def augment_image(self, image):
        width, height = image.size

        # ---------- 1. 基础几何变换 ----------
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() > 0.5:
            angle = random.randint(-45, 45)
            image = image.rotate(angle, resample=Image.BICUBIC, expand=False)

        # ---------- 2. 透视变换 ----------
        if random.random() > 0.6:
            img_np = np.array(image)
            distortion_scale = random.uniform(0.1, 0.3)
            half_width = width // 2
            half_height = height // 2
            topleft = (random.randint(0, int(distortion_scale * half_width)),
                       random.randint(0, int(distortion_scale * half_height)))
            topright = (width - random.randint(0, int(distortion_scale * half_width)),
                        random.randint(0, int(distortion_scale * half_height)))
            botright = (width - random.randint(0, int(distortion_scale * half_width)),
                        height - random.randint(0, int(distortion_scale * half_height)))
            botleft = (random.randint(0, int(distortion_scale * half_width)),
                       height - random.randint(0, int(distortion_scale * half_height)))
            from_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            to_points = np.float32([topleft, topright, botright, botleft])
            M = cv2.getPerspectiveTransform(from_points, to_points)
            img_np = cv2.warpPerspective(img_np, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            image = Image.fromarray(img_np)

        # ---------- 3. 高斯噪声（保持二值特性） ----------
        if random.random() > 0.8:
            img_np = np.array(image).astype(np.float32)
            noise = np.random.normal(0, random.uniform(10, 30), img_np.shape)
            img_np = img_np + noise
            img_np = np.clip(img_np, 0, 255)
            img_np = (img_np > 128).astype(np.uint8) * 255
            image = Image.fromarray(img_np.astype(np.uint8))

        # ---------- 4. 模糊 ----------
        if random.random() > 0.8:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))

        # ---------- 5. 缩放和填充 ----------
        if random.random() > 0.5:
            scale_factor = random.uniform(0.8, 1.2)
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

        # ---------- 6. 随机擦除（全黑块） ----------
        if random.random() > 0.8:
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            for _ in range(random.randint(1, 3)):
                erase_size_h = random.randint(h // 20, h // 10)
                erase_size_w = random.randint(w // 20, w // 10)
                y = random.randint(0, h - erase_size_h)
                x = random.randint(0, w - erase_size_w)
                img_np[y:y + erase_size_h, x:x + erase_size_w] = 0
            image = Image.fromarray(img_np)

        # ---------- 7. 随机抹黑/抹白/“补块” ----------
        if random.random() > 0.85:
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            num_patches = random.randint(1, 4)
            for _ in range(num_patches):
                erase_h = random.randint(h // 20, h // 8)
                erase_w = random.randint(w // 20, w // 8)
                y = random.randint(0, h - erase_h)
                x = random.randint(0, w - erase_w)
                fill_value = 0 if random.random() > 0.5 else 255  # 随机补黑/补白
                img_np[y:y + erase_h, x:x + erase_w] = fill_value
            image = Image.fromarray(img_np)

        # ---------- 8. 形态学扰动：随机开操作（去除细须）或闭操作（填小洞） ----------
        if random.random() > 0.5:
            img_np = np.array(image)
            kernel_size = random.choice([3, 5, 7, 9])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            if random.random() > 0.5:
                img_np = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)  # 去须、清小点
            else:
                img_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)  # 填小洞
            image = Image.fromarray(img_np)

        # ---------- 9. （可选）形态学腐蚀/膨胀，对比度增强 ----------
        if random.random() > 0.5:
            img_np = np.array(image)
            kernel_size = random.choice([3, 5])
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            if random.random() > 0.5:
                img_np = cv2.erode(img_np, kernel, iterations=1)
            else:
                img_np = cv2.dilate(img_np, kernel, iterations=1)
            image = Image.fromarray(img_np)

        # 保证输出依然是“二值图三通道”
        image = image.convert("L").convert("RGB")
        return image

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image']
        img = Image.open(img_path).convert("L").convert("RGB")
        if self.use_augment:
            img1 = self.augment_image(img)
            img2 = self.augment_image(img)
        else:
            img1 = img
            img2 = img
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2
