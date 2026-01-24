import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
IMAGE_ROOT = "./StitchingNet-seg/dataset"
IMAGESETS_DIR = "./StitchingNet-seg/splits"
EXCLUDED_CLASSES = [6, 8, 9]
IMG_EXTENSIONS = ['.jpg', '.png']

# Mapping from Dataset Folder ID (Defect Type) to Segmentation Class ID
# Format: {Folder_ID: Segmentation_Class_ID}
SEG_CLASS_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 6, 10: 7}
NUM_CLASSES = len(SEG_CLASS_MAP) + 1  # +1 for Background

def get_file_paths():
    """
    Traverses the dataset directory to collect image and mask paths.
    Returns lists of metadata for train, validation, and test sets.
    """
    material_folders = sorted([d for d in os.listdir(IMAGE_ROOT) if os.path.isdir(os.path.join(IMAGE_ROOT, d))])
    
    all_image_meta = {}

    for material_folder in material_folders:
        material_path = os.path.join(IMAGE_ROOT, material_folder)

        for class_folder in os.listdir(material_path):
            try:
                # Folder name indicates the defect type ID
                class_id = int(class_folder.split('.')[0])
            except ValueError:
                continue
            
            if class_id in EXCLUDED_CLASSES:
                continue
            
            class_path_base = os.path.join(material_path, class_folder)
            image_dir = os.path.join(class_path_base, 'image')
            mask_dir = os.path.join(class_path_base, 'mask')

            if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
                continue

            for filename in os.listdir(image_dir):
                basename, ext = os.path.splitext(filename)
                if ext.lower() not in IMG_EXTENSIONS:
                    continue

                img_path = os.path.join(image_dir, filename)
                mask_path = os.path.join(mask_dir, basename + '.png')

                if not os.path.exists(mask_path):
                    continue
                
                # Store metadata: (image_path, mask_path, defect_type_id)
                all_image_meta[basename] = (img_path, mask_path, class_id)

    def load_data_from_split_file(split_filename):
        """
        Loads file paths based on the text files in the splits directory.
        """
        split_path = os.path.join(IMAGESETS_DIR, split_filename)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")
            
        data_list = []
        with open(split_path, 'r') as f:
            basenames = [line.strip() for line in f if line.strip()]
            
        for basename in basenames:
            meta = all_image_meta.get(basename)
            if meta:
                data_list.append(meta)
                
        return data_list

    train_data = load_data_from_split_file("train.txt")
    val_data = load_data_from_split_file("val.txt")
    test_data = load_data_from_split_file("test.txt")

    return train_data, val_data, test_data


class SegmentationDataset(Dataset):
    """
    Custom Dataset for Semantic Segmentation.
    """
    def __init__(self, file_data, seg_map, transforms=None):
        self.file_data = file_data
        self.seg_map = seg_map
        self.transforms = transforms

    def __len__(self):
        return len(self.file_data)

    def __getitem__(self, idx):
        img_path, mask_path, original_defect_id = self.file_data[idx]
        
        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to read mask file: {mask_path}")
            
        # Generate Segmentation Mask based on Defect ID
        seg_mask = np.zeros_like(mask, dtype=np.int64)
        
        if original_defect_id != 0: 
            target_seg_id = self.seg_map[original_defect_id]
            seg_mask[mask > 0] = target_seg_id
        
        if self.transforms:
            transformed = self.transforms(image=image, masks=[seg_mask])
            image = transformed['image']
            seg_mask = transformed['masks'][0]
            
        return image, seg_mask.long()
    
def get_transforms(image_size=256):
    """
    Returns Albumentations transforms for training and validation.
    """
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_test_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_transform, val_test_transform

def seed_worker(worker_id):
    """
    Ensures reproducibility in data loading.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloaders(batch_size=8, image_size=256, num_workers=4, seed=42):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    train_data, val_data, test_data = get_file_paths()
    train_transform, val_test_transform = get_transforms(image_size)

    train_dataset = SegmentationDataset(train_data, SEG_CLASS_MAP, transforms=train_transform)
    val_dataset = SegmentationDataset(val_data, SEG_CLASS_MAP, transforms=val_test_transform)
    test_dataset = SegmentationDataset(test_data, SEG_CLASS_MAP, transforms=val_test_transform)

    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, 
        worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, 
        worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, 
        worker_init_fn=seed_worker
    )
    
    return train_loader, val_loader, test_loader, NUM_CLASSES

if __name__ == '__main__':
    print("Initializing DataLoaders...")
    train_dl, val_dl, test_dl, num_classes = get_dataloaders(batch_size=4, num_workers=0)
    
    print(f"Number of Segmentation Classes: {num_classes}")
    print(f"Train batches: {len(train_dl)}")
    
    if len(train_dl) > 0:
        print("\nChecking first batch...")
        images, seg_masks = next(iter(train_dl))
        
        print(f"Image shape: {images.shape}")
        print(f"Mask shape: {seg_masks.shape}")
        print(f"Unique mask values: {torch.unique(seg_masks)}")
    else:
        print("\nError: Train dataset is empty.")