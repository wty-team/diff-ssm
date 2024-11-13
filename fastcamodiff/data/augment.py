import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class CamouflageAugmentation:
    def __init__(self, config):
        self.config = config

        # 基础增强概率
        self.flip_prob = 0.5
        self.rotate_prob = 0.3
        self.color_prob = 0.3
        self.blur_prob = 0.2
        self.noise_prob = 0.2

        # 色彩增强范围
        self.brightness_factor = (0.8, 1.2)
        self.contrast_factor = (0.8, 1.2)
        self.saturation_factor = (0.8, 1.2)
        self.hue_factor = (-0.1, 0.1)

        # 旋转角度范围
        self.rotation_range = (-30, 30)

        # 归一化参数
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, image, mask):
        # 确保输入是PIL图像
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(mask)

        # 随机水平翻转
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 随机垂直翻转
        if random.random() < self.flip_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # 随机旋转
        if random.random() < self.rotate_prob:
            angle = random.uniform(*self.rotation_range)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # 色彩增强 (只对图像进行)
        if random.random() < self.color_prob:
            # 亮度
            brightness_factor = random.uniform(*self.brightness_factor)
            image = ImageEnhance.Brightness(image).enhance(brightness_factor)

            # 对比度
            contrast_factor = random.uniform(*self.contrast_factor)
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)

            # 饱和度
            saturation_factor = random.uniform(*self.saturation_factor)
            image = ImageEnhance.Color(image).enhance(saturation_factor)

        # 随机模糊
        if random.random() < self.blur_prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))

        # 随机噪声
        if random.random() < self.noise_prob:
            image_np = np.array(image)
            noise = np.random.normal(0, 25, image_np.shape)
            image_np = np.clip(image_np + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(image_np)

        # 转换为tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # 归一化图像
        image = self.normalize(image)

        return image, mask


class TestTimeAugmentation:
    def __init__(self, config):
        self.config = config
        self.num_augmentations = 4  # TTA使用的增强数量

        # 归一化参数
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def augment_image(self, image, aug_type):
        """对单张图像进行测试时增强"""
        if aug_type == 0:  # 原始图像
            return image
        elif aug_type == 1:  # 水平翻转
            return TF.hflip(image)
        elif aug_type == 2:  # 垂直翻转
            return TF.vflip(image)
        elif aug_type == 3:  # 旋转90度
            return TF.rotate(image, 90)

    def reverse_augment(self, pred, aug_type):
        """反转增强操作"""
        if aug_type == 0:  # 原始图像
            return pred
        elif aug_type == 1:  # 水平翻转
            return TF.hflip(pred)
        elif aug_type == 2:  # 垂直翻转
            return TF.vflip(pred)
        elif aug_type == 3:  # 旋转-90度
            return TF.rotate(pred, -90)

    def __call__(self, model, image):
        """
        对输入图像进行测试时增强,并整合所有预测结果
        Args:
            model: 训练好的模型
            image: PIL Image对象
        Returns:
            merged_pred: 合并后的预测掩码
        """
        # 确保输入是PIL图像
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # 所有预测结果
        predictions = []

        for aug_idx in range(self.num_augmentations):
            # 应用增强
            aug_image = self.augment_image(image, aug_idx)

            # 转换为tensor并归一化
            aug_tensor = self.normalize(TF.to_tensor(aug_image))
            aug_tensor = aug_tensor.unsqueeze(0)  # 添加batch维度

            # 获取预测结果
            with torch.no_grad():
                pred = model(aug_tensor)

            # 转换预测结果为图像格式
            pred = pred.squeeze().cpu()
            pred = torch.sigmoid(pred)  # 如果模型输出未经过sigmoid

            # 反转增强操作
            pred = self.reverse_augment(pred, aug_idx)

            predictions.append(pred)

        # 合并所有预测结果(平均)
        merged_pred = torch.stack(predictions).mean(0)

        return merged_pred


def get_augmentation(config, is_train=True):
    """
    工厂函数,根据配置返回相应的增强器
    """
    if is_train:
        return CamouflageAugmentation(config)
    else:
        return TestTimeAugmentation(config)