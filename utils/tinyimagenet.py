import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.root = root
        self.train = train
        self.transform = transform
        self.train_dir = os.path.join(root, 'tiny-imagenet-200', 'train')
        self.val_dir = os.path.join(root, 'tiny-imagenet-200', 'val')
        
        if download:
            self._download()
        
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
        
        # Load classes
        self.classes = []
        with open(os.path.join(root, 'tiny-imagenet-200', 'wnids.txt'), 'r') as f:
            for line in f:
                self.classes.append(line.strip())
        
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load data
        if self.train:
            self.data, self.targets = self._load_train_data()
        else:
            self.data, self.targets = self._load_val_data()
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'tiny-imagenet-200'))
    
    def _download(self):
        if self._check_exists():
            return
        
        os.makedirs(self.root, exist_ok=True)
        print('Downloading TinyImageNet dataset...')
        print('Please download the dataset manually from http://cs231n.stanford.edu/tiny-imagenet-200.zip')
        print('Extract it to {}'.format(self.root))
        print('The directory structure should be:')
        print('  {}/tiny-imagenet-200/'.format(self.root))
        print('  {}/tiny-imagenet-200/train/'.format(self.root))
        print('  {}/tiny-imagenet-200/val/'.format(self.root))
        print('  {}/tiny-imagenet-200/test/'.format(self.root))
    
    def _load_train_data(self):
        data = []
        targets = []
        
        for class_dir in os.listdir(self.train_dir):
            class_idx = self.class_to_idx[class_dir]
            images_dir = os.path.join(self.train_dir, class_dir, 'images')
            
            for img_file in os.listdir(images_dir):
                if img_file.endswith('.JPEG'):
                    img_path = os.path.join(images_dir, img_file)
                    data.append(img_path)
                    targets.append(class_idx)
        
        return data, targets
    
    def _load_val_data(self):
        data = []
        targets = []
        
        val_annotations_path = os.path.join(self.root, 'tiny-imagenet-200', 'val', 'val_annotations.txt')
        
        with open(val_annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_file, class_id = parts[0], parts[1]
                
                if class_id in self.class_to_idx:
                    img_path = os.path.join(self.val_dir, 'images', img_file)
                    data.append(img_path)
                    targets.append(self.class_to_idx[class_id])
        
        return data, targets
    
    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        return len(self.data)
