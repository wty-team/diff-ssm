import os
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms


class CamouflageDataset(Dataset):
    def __init__(self, config, dataset_name, is_train=True):
        """
        Args:
            config: Configuration object
            dataset_name: Name of dataset ('camo', 'cod10k', or 'nc4k')
            is_train: Whether this is training set
        """
        self.config = config
        self.dataset_name = dataset_name
        self.is_train = is_train

        # Get root directory
        self.root_dir = config.train_data[dataset_name] if is_train else config.val_data[dataset_name]

        # Get image and mask directories
        self.image_dir = os.path.join(self.root_dir,
                                      config.data_structure[dataset_name]['image'])
        self.mask_dir = os.path.join(self.root_dir,
                                     config.data_structure[dataset_name]['mask'])

        # Get all image paths
        self.image_paths = []
        self.mask_paths = []
        for img_name in os.listdir(self.image_dir):
            if img_name.endswith(('.jpg', '.png')):
                img_path = os.path.join(self.image_dir, img_name)
                mask_name = img_name.replace('.jpg', '.png')
                mask_path = os.path.join(self.mask_dir, mask_name)

                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)

        # 设置数据增强
        self.transform = self._get_transform()

    def _get_transform(self):
        if self.is_train:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)

        return {
            'image': image,
            'mask': mask,
            'path': self.image_paths[idx]
        }


def get_dataloader(config, is_train=True):
    """Create combined dataloader from all datasets"""
    datasets = []

    # Create dataset for each source
    for dataset_name in ['camo', 'cod10k', 'nc4k']:
        dataset = CamouflageDataset(config, dataset_name, is_train)
        datasets.append(dataset)

    # Combine datasets
    combined_dataset = ConcatDataset(datasets)

    return torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=config.batch_size,
        shuffle=is_train,
        num_workers=config.num_workers,
        pin_memory=True
    )