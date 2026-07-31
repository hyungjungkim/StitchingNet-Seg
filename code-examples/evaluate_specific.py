# Part 3. Evaluate specifically

import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Image Path
IMAGE_ROOT = "content/StitchingNet-Seg/dataset"
IMAGESETS_DIR = "content/StitchingNet-Seg/splits"

# Model & Result Path
BASE_DIR = "./result_models"
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "trained_models")
RESULT_SAVE_DIR = os.path.join(BASE_DIR, "benchmark_results_specific")
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

# Hyperparameters & Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_WORKERS = 4
SEED = 64

# Dataset Settings
EXCLUDED_CLASSES = [6, 8, 9]
IMG_EXTENSIONS = ['.jpg', '.png']
SEG_CLASS_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 6, 10: 7}
NUM_CLASSES = len(SEG_CLASS_MAP) + 1  

DEFECT_CLASSES = [
    "Skipped stitch",
    "Broken stitch",
    "Pinched fabric",
    "Crooked seam",
    "Thread sagging",
    "Stain and damage",
    "Overlapped stitch"
]

# Dataset Preparation
def get_file_paths():
    """
    Traverses the dataset directory to collect image paths, mask paths, and fabric names.
    Since this is an evaluation script, it focuses primarily on retrieving test metadata.
    """
    fabric_folders = sorted([d for d in os.listdir(IMAGE_ROOT) if os.path.isdir(os.path.join(IMAGE_ROOT, d))])
    all_image_meta = {}

    for fabric_folder in fabric_folders:
        fabric_path = os.path.join(IMAGE_ROOT, fabric_folder)

        for class_folder in os.listdir(fabric_path):
            try:
                class_id = int(class_folder.split('.')[0])
            except ValueError:
                continue

            if class_id in EXCLUDED_CLASSES:
                continue

            class_path_base = os.path.join(fabric_path, class_folder)
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

                # Store metadata: (image_path, mask_path, defective_class_id, fabric_name)
                all_image_meta[basename] = (img_path, mask_path, class_id, fabric_folder)


    def load_data_from_split_file(split_filename):
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

    test_data = load_data_from_split_file("test.txt")

    return None, None, test_data


class SegmentationDataset(Dataset):
    def __init__(self, file_data, seg_map, transforms=None):
        self.file_data = file_data
        self.seg_map = seg_map
        self.transforms = transforms

    def __len__(self):
        return len(self.file_data)

    def __getitem__(self, idx):
        img_path, mask_path, original_defect_id, fabric_name = self.file_data[idx]

        # Load image & mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to read mask file: {mask_path}")

        # Generate segmentation mask based on mapped defect ID
        seg_mask = np.zeros_like(mask, dtype=np.int64)
        if original_defect_id != 0:
            target_seg_id = self.seg_map[original_defect_id]
            seg_mask[mask > 0] = target_seg_id

        if self.transforms:
            transformed = self.transforms(image=image, masks=[seg_mask])
            image = transformed['image']
            seg_mask = transformed['masks'][0]

        return image, seg_mask.long(), fabric_name


def get_transforms(image_size=256):
    val_test_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return val_test_transform


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(batch_size=8, image_size=256, num_workers=4, seed=42):
    """
    Creates and returns only the test DataLoader to optimize memory usage 
    since this script is strictly for benchmarking/evaluation.
    """
    _, _, test_data = get_file_paths()
    val_test_transform = get_transforms(image_size)

    test_dataset = SegmentationDataset(test_data, SEG_CLASS_MAP, transforms=val_test_transform)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=seed_worker
    )

    return None, None, test_loader, NUM_CLASSES


# Model
class ModelFactory(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model_name = model_name
        if model_name == 'ResNet-UNet':
            self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=num_classes)
        elif model_name == 'UNet++':
            self.model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=num_classes)
        elif model_name == 'DeepLabV3':
            self.model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=num_classes)
        elif model_name == 'SegFormer':
            self.model = smp.Segformer(encoder_name="mit_b2", encoder_weights=None, in_channels=3, classes=num_classes)
        elif model_name == 'Swin-Unet':
            self.model = smp.Unet(encoder_name="tu-swin_tiny_patch4_window7_224", encoder_weights=None, in_channels=3, classes=num_classes)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def forward(self, x):
        return self.model(x)

def _fast_hist(label, pred, n_class):
    """
    Computes the confusion matrix for a given predicted and ground-truth mask.
    """
    mask = (label >= 0) & (label < n_class)
    hist = np.bincount(
        n_class * label[mask].astype(int) + pred[mask],
        minlength=n_class ** 2
    ).reshape(n_class, n_class)
    return hist

def calculate_per_class_metrics(hist):
    """
    Calculates per-class IoU and Dice scores based on the confusion matrix.
    """
    epsilon = 1e-7
    tp = np.diag(hist)
    fp = hist.sum(axis=0) - tp
    fn = hist.sum(axis=1) - tp

    iou = tp / (tp + fp + fn + epsilon)
    dice = 2 * tp / (2 * tp + fp + fn + epsilon)

    return iou, dice


# Evaluation
def evaluate_and_save_csv(model_name, test_dl, num_classes):
    print(f"\nEvaluating [ {model_name} ] for fabric-defect combinations")
    
    model = ModelFactory(model_name, num_classes).to(DEVICE)
    weights_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_best.pth")
    
    if not os.path.exists(weights_path):
        print(f"Weights not found for {model_name} at {weights_path}. Skipping.")
        return

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    # Dictionary to store Confusion Matrix per fabric type
    hist_dict = {}

    with torch.no_grad():
        for batch_data in tqdm(test_dl, desc=f"Inference ({model_name})"):
            if len(batch_data) == 3:
                imgs, masks, fabrics = batch_data
            else:
                raise ValueError("Dataloader must yield (imgs, masks, fabrics) for fabric-level evaluation.")

            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            
            if outputs.shape[-2:] != imgs.shape[-2:]:
                outputs = nn.functional.interpolate(outputs, size=imgs.shape[-2:], mode='bilinear', align_corners=False)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.numpy()
            
            # Accumulate histogram based on fabric info for each image in the batch
            for i in range(len(imgs)):
                fabric_name = str(fabrics[i])
                if fabric_name not in hist_dict:
                    hist_dict[fabric_name] = np.zeros((num_classes, num_classes))
                
                hist_dict[fabric_name] += _fast_hist(masks[i].flatten(), preds[i].flatten(), num_classes)

    results_data = []
    
    # Record metrics per fabric and defect type
    for fabric_name in sorted(hist_dict.keys()):
        fabric_hist = hist_dict[fabric_name]
        iou_array, dice_array = calculate_per_class_metrics(fabric_hist)
        
        for i, defect_name in enumerate(DEFECT_CLASSES, start=1):
            iou_val = iou_array[i] * 100
            dice_val = dice_array[i] * 100
            
            results_data.append({
                "Fabric": fabric_name,
                "Defect class": defect_name,
                "mIoU (%)": round(iou_val, 2),
                "mDice (%)": round(dice_val, 2)
            })
            
        # Calculate the mean for all defects in the current fabric (excluding background, classes 1-7)
        fab_iou_mean = np.nanmean(iou_array[1:len(DEFECT_CLASSES)+1]) * 100
        fab_dice_mean = np.nanmean(dice_array[1:len(DEFECT_CLASSES)+1]) * 100
        
        results_data.append({
            "Fabric": fabric_name,
            "Defect class": "Average (All Defects)",
            "mIoU (%)": round(fab_iou_mean, 2),
            "mDice (%)": round(fab_dice_mean, 2)
        })

    # Aggregate all histograms to compute the overall dataset average
    total_hist = sum(hist_dict.values())
    total_iou, total_dice = calculate_per_class_metrics(total_hist)
    
    for i, defect_name in enumerate(DEFECT_CLASSES, start=1):
        results_data.append({
            "Fabric": "TOTAL (All Fabrics)",
            "Defect class": defect_name,
            "mIoU (%)": round(total_iou[i] * 100, 2),
            "mDice (%)": round(total_dice[i] * 100, 2)
        })
        
    total_iou_mean = np.nanmean(total_iou[1:len(DEFECT_CLASSES)+1]) * 100
    total_dice_mean = np.nanmean(total_dice[1:len(DEFECT_CLASSES)+1]) * 100
    results_data.append({
        "Fabric": "TOTAL (All Fabrics)",
        "Defect class": "Average (All Defects)",
        "mIoU (%)": round(total_iou_mean, 2),
        "mDice (%)": round(total_dice_mean, 2)
    })

    df = pd.DataFrame(results_data)
    df["mIoU (%)"] = df["mIoU (%)"].map(lambda x: f"{x:.2f}")
    df["mDice (%)"] = df["mDice (%)"].map(lambda x: f"{x:.2f}")

    # Save to CSV
    csv_filename = os.path.join(RESULT_SAVE_DIR, f"{model_name}_fabric_defect_evaluation.csv")
    df.to_csv(csv_filename, index=False)
    
    print(f"Saved detailed evaluation for {model_name} to: {csv_filename}\n")


# Main
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    print("Initializing DataLoader for Evaluation")
    _, _, test_dl, num_classes = get_dataloaders(
        batch_size=BATCH_SIZE, 
        image_size=IMAGE_SIZE, 
        num_workers=NUM_WORKERS, 
        seed=SEED
    )

    print(f"Number of segmentation classes: {num_classes}")
    
    # List of models to evaluate sequentially
    target_models = ['ResNet-UNet', 'UNet++', 'DeepLabV3', 'SegFormer', 'Swin-Unet']
    
    for model_name in target_models:
        evaluate_and_save_csv(model_name, test_dl, num_classes)