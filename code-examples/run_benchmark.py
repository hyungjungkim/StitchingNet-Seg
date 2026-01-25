# Part 2. Run benchmark

import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
import os
import sys
import pandas as pd
import random
import matplotlib.pyplot as plt

# Configuration & hyperparameters
BASE_DIR = "./result_models"
sys.path.append(BASE_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
IMAGE_SIZE = 224
PATIENCE = 8
SEED = 64
NUM_WORKERS = 4

# Directory setup
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "trained_models")
RESULT_SAVE_DIR = os.path.join(BASE_DIR, "benchmark_results")
VIS_SAVE_DIR = os.path.join(BASE_DIR, "visual_results")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)
os.makedirs(VIS_SAVE_DIR, exist_ok=True)

# Color map for visualization (fixed colors for consistency)
# Format: [R, G, B]
COLOR_MAP = [
    [0, 0, 0],       # Background
    [255, 0, 0],     # Class 1
    [0, 255, 0],     # Class 2
    [0, 0, 255],     # Class 3
    [255, 255, 0],   # Class 4
    [255, 0, 255],   # Class 5
    [0, 255, 255],   # Class 6
    [128, 0, 0],     # Class 7
    [0, 128, 0],     # Class 8
    [0, 0, 128],     # Class 9
]


# Model
class ModelFactory(nn.Module):
    """
    Factory class to instantiate various segmentation models using SMP.
    """
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model_name = model_name

        if model_name == 'ResUNet':
            self.model = smp.Unet(
                encoder_name="resnet34", encoder_weights="imagenet",
                in_channels=3, classes=num_classes
            )
        elif model_name == 'UNet++':
            self.model = smp.UnetPlusPlus(
                encoder_name="resnet34", encoder_weights="imagenet",
                in_channels=3, classes=num_classes
            )
        elif model_name == 'DeepLabV3':
            self.model = smp.DeepLabV3Plus(
                encoder_name="resnet34", encoder_weights="imagenet",
                in_channels=3, classes=num_classes
            )
        elif model_name == 'SegFormer':
            self.model = smp.Segformer(
                encoder_name="mit_b2", encoder_weights="imagenet",
                in_channels=3, classes=num_classes
            )
        elif model_name == 'Swin-Unet':
            self.model = smp.Unet(
                encoder_name="tu-swin_tiny_patch4_window7_224", encoder_weights="imagenet",
                in_channels=3, classes=num_classes
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def forward(self, x):
            return self.model(x)


# Utility functions
def set_seed(seed=42):
    """
    Sets random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def _fast_hist(label, pred, n_class):
    """
    Calculates the confusion matrix for segmentation.
    """
    mask = (label >= 0) & (label < n_class)
    hist = np.bincount(
        n_class * label[mask].astype(int) + pred[mask],
        minlength=n_class ** 2
    ).reshape(n_class, n_class)
    return hist

def calculate_metrics(hist):
    """
    Computes mIoU, mDice, and Recall from the confusion matrix.
    """
    epsilon = 1e-7
    tp = np.diag(hist)
    fp = hist.sum(axis=0) - tp
    fn = hist.sum(axis=1) - tp

    iou = tp / (tp + fp + fn + epsilon)
    dice = 2 * tp / (2 * tp + fp + fn + epsilon)
    recall = tp / (tp + fn + epsilon)

    miou = np.nanmean(iou)
    mdice = np.nanmean(dice)
    mrecall = np.nanmean(recall)

    return miou, mdice, mrecall

def colorize_mask(mask):
    """
    Converts a (H, W) label mask into a (H, W, 3) RGB image for visualization.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(COLOR_MAP):
        img[mask == class_id] = color

    return img

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalizes a tensor image back to the [0, 255] range.
    """
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def save_visual_results(model_name, images, gt_masks, pred_masks, batch_idx, save_dir):
    """
    Saves a comparison of Original, GT, and Prediction images.
    """
    os.makedirs(save_dir, exist_ok=True)
    batch_size = images.size(0)

    for i in range(batch_size):
        orig_img = denormalize(images[i])
        gt_color = colorize_mask(gt_masks[i])
        pred_color = colorize_mask(pred_masks[i])
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        ax[0].imshow(orig_img)
        ax[0].set_title(f"Original")
        ax[0].axis('off')

        ax[1].imshow(gt_color)
        ax[1].set_title("Ground Truth")
        ax[1].axis('off')

        ax[2].imshow(pred_color)
        ax[2].set_title(f"{model_name} Prediction")
        ax[2].axis('off')

        file_name = f"batch{batch_idx:02d}_img{i:03d}_{model_name}.png"

        save_path = os.path.join(save_dir, file_name)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)


# Training & evaluation
def run_experiment(model_name, train_dl, val_dl, test_dl, num_classes):
    print(f"\n" + "="*60)
    print(f"Start Training: {model_name}")
    print("="*60)

    model = ModelFactory(model_name, num_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Loss functions
    loss_dice = smp.losses.DiceLoss(mode='multiclass')
    loss_ce = nn.CrossEntropyLoss()

    best_miou = 0.0
    save_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_best.pth")
    epochs_no_improve = 0

    # Training
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_dl, desc=f"[{model_name}] Ep {epoch+1}/{EPOCHS}", leave=False)

        for imgs, masks in loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(imgs)

            # Interpolate if output size mismatch
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            loss = loss_dice(outputs, masks) + loss_ce(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        hist = np.zeros((num_classes, num_classes))

        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs = imgs.to(DEVICE)
                masks = masks.numpy()

                outputs = model(imgs)
                if outputs.shape[-2:] != imgs.shape[-2:]:
                    outputs = nn.functional.interpolate(outputs, size=imgs.shape[-2:], mode='bilinear', align_corners=False)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                hist += _fast_hist(masks.flatten(), preds.flatten(), num_classes)

        miou, mdice, _ = calculate_metrics(hist)
        print(f"[Ep {epoch+1}] Val -> Train_loss: {train_loss/len(train_dl):.4f} | mIoU: {miou:.4f} | mDice: {mdice:.4f}")

        # Save best model
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            print(f"   -> Best model saved! (mIoU: {best_miou:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"   -> Early stopping triggered at epoch {epoch+1}.")
                break

    # Final benchmark (load best model & visualize)
    print(f"\nEvaluating best {model_name} on the test dataset...")
    model.load_state_dict(torch.load(save_path))
    model.eval()

    model_vis_dir = os.path.join(VIS_SAVE_DIR, model_name)
    os.makedirs(model_vis_dir, exist_ok=True)

    hist = np.zeros((num_classes, num_classes))
    save_count = 0
    MAX_SAVE_BATCHES = 5

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(tqdm(test_dl, desc="Testing")):
            imgs = imgs.to(DEVICE)

            outputs = model(imgs)
            if outputs.shape[-2:] != imgs.shape[-2:]:
                outputs = nn.functional.interpolate(outputs, size=imgs.shape[-2:], mode='bilinear', align_corners=False)

            preds = torch.argmax(outputs, dim=1)

            gt_numpy = masks.numpy()
            pred_numpy = preds.cpu().numpy()
            hist += _fast_hist(gt_numpy.flatten(), pred_numpy.flatten(), num_classes)

            if save_count < MAX_SAVE_BATCHES:
                save_visual_results(model_name, imgs, masks, preds, batch_idx, model_vis_dir)
                save_count += 1

    test_miou, test_mdice, test_recall = calculate_metrics(hist)
    print(f"Final benchmark result ({model_name}) -> mIoU: {test_miou:.4f}, Dice: {test_mdice:.4f}")
    print(f"Visual results saved to: {model_vis_dir}")

    return {
        'Model': model_name,
        'mIoU': test_miou,
        'mDice': test_mdice,
        'Recall': test_recall
    }


# Run benchmark
set_seed(SEED)
print(f"The experiment for model benchmark is started on {DEVICE}")
print(f"Base directory: {BASE_DIR}")

# 1. Load StitchingNet-Seg dataset
train_dl, val_dl, test_dl, num_classes = get_dataloaders(
    batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=NUM_WORKERS, seed=SEED
)

# 2. Set target models to benchmark
# target_models = ['ResUNet', 'UNet++', 'DeepLabV3', 'SegFormer', 'Swin-Unet']
target_models = ['ResUNet', 'SegFormer']
results = []

# 3. Run experiments
for model_name in target_models:
    set_seed(SEED)
    try:
        res = run_experiment(model_name, train_dl, val_dl, test_dl, num_classes)
        results.append(res)
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        import traceback
        traceback.print_exc()

# 4. Save final benchmark results
if results:
    df = pd.DataFrame(results)
    df = df.sort_values(by='mIoU', ascending=False)

    save_path = os.path.join(RESULT_SAVE_DIR, "final_benchmark_results.csv")
    df.to_csv(save_path, index=False)

    print("\n" + "="*60)
    print("Final benchmark results")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\nResults saved to: {save_path}")
else:
    print("No results to save.")