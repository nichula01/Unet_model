# =======================================================================================
#
#  COMPLETE SLUM DETECTION PIPELINE (V4 - FINAL WITH ADVANCED METRICS & UI SUPPORT)
#
# =======================================================================================

import os
import sys
import glob
import random
import time
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd
import cv2
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset, DataLoader

import albumentations as albu
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import rasterio
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")

# =======================================================================================
# 1. CONFIGURATION PARAMETERS
# =======================================================================================
class Config:
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    TILES_DIR = os.path.join(DATA_DIR, "segmentation_tiles")
    IMAGE_TILES_DIR = os.path.join(TILES_DIR, "images")
    MASK_TILES_DIR = os.path.join(TILES_DIR, "masks")
    
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    IMAGESIZE = 256
    CLASSES = 3
    CLASS_NAMES = ['Background', 'Urban', 'Slum']
    
    MODEL_NAME = "UNet"
    TARGET_CITY = "mumbai"
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    
    SEED = 42

config = Config()

for path in [config.DATA_DIR, config.RESULTS_DIR, config.TILES_DIR, config.IMAGE_TILES_DIR, config.MASK_TILES_DIR, config.MODEL_DIR]:
    os.makedirs(path, exist_ok=True)

# =======================================================================================
# 2. HELPER FUNCTIONS AND DEVICE SETUP
# =======================================================================================
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

DEVICE = get_device()

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config.SEED)

# =======================================================================================
# 3. U-NET MODEL ARCHITECTURE
# =======================================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128); self.down2 = Down(128, 256);
        self.down3 = Down(256, 512); self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512); self.up2 = Up(512, 256)
        self.up3 = Up(256, 128); self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3)
        x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# =======================================================================================
# 4. DATA PIPELINE & AUGMENTATIONS
# =======================================================================================
def split_raster_function(city_name, data_dir, image_tiles_dir, mask_tiles_dir):
    print(f"--- Starting Raster Splitting for {city_name} ---")
    try:
        image_path = os.path.join(data_dir, f"{city_name}_3m.tif")
        slum_ref_path = os.path.join(data_dir, f"{city_name}_slum_reference.tif")
        urban_ref_path = os.path.join(data_dir, f"{city_name}_urban_reference.tif")
        image = tifffile.imread(image_path)[:, :, 0:3]
        slum_mask = tifffile.imread(slum_ref_path)
        urban_mask = tifffile.imread(urban_ref_path)
    except FileNotFoundError as e:
        print(f"Error: Missing a required source file. {e}"); return

    final_mask = np.zeros_like(slum_mask, dtype=np.uint8)
    final_mask[urban_mask == 1] = 1
    final_mask[slum_mask == 1] = 2
    size = config.IMAGESIZE
    stride = size // 2
    tile_id = 0
    for r in tqdm(range(0, image.shape[0] - size + 1, stride), desc="Tiling..."):
        for c in range(0, image.shape[1] - size + 1, stride):
            img_tile = image[r:r+size, c:c+size]
            mask_tile = final_mask[r:r+size, c:c+size]
            tifffile.imwrite(os.path.join(image_tiles_dir, f"{city_name}_{tile_id}.tif"), img_tile)
            tifffile.imwrite(os.path.join(mask_tiles_dir, f"{city_name}_{tile_id}.tif"), mask_tile)
            tile_id += 1
    print(f"Finished splitting. Created {tile_id} image/mask pairs.")

class SegmentationDataset(BaseDataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths, self.mask_paths, self.transform = image_paths, mask_paths, transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        image = tifffile.imread(self.image_paths[idx])
        mask = tifffile.imread(self.mask_paths[idx])
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return image, mask.long()

def get_transforms():
    train_transform = albu.Compose([
        albu.HorizontalFlip(p=0.5), albu.VerticalFlip(p=0.5), albu.RandomRotate90(p=0.5),
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2(),
    ])
    val_transform = albu.Compose([
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2(),
    ])
    return train_transform, val_transform

# =======================================================================================
# 5. TRAINING, METRICS, AND INFERENCE
# =======================================================================================
def iou_score(pred, target, n_classes):
    pred = torch.argmax(pred, dim=1)
    iou = 0.0
    for cls in range(n_classes):
        pred_inds, target_inds = (pred == cls), (target == cls)
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        iou += (intersection / union) if union != 0 else 1.0
    return iou / n_classes

def train_one_epoch(loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss, total_iou = 0, 0
    for images, masks in loop:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        predictions = model(images)
        loss = loss_fn(predictions, masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        iou = iou_score(predictions, masks, config.CLASSES)
        total_loss += loss.item(); total_iou += iou
        loop.set_postfix(loss=loss.item(), iou=iou)
    return total_loss / len(loader), total_iou / len(loader)

def evaluate_model(loader, model, loss_fn):
    model.eval()
    loop = tqdm(loader, desc="Validating")
    total_loss, total_iou = 0, 0
    all_preds, all_gts = [], []
    with torch.no_grad():
        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            predictions = model(images)
            loss = loss_fn(predictions, masks)
            iou = iou_score(predictions, masks, config.CLASSES)
            total_loss += loss.item(); total_iou += iou
            all_preds.append(torch.argmax(predictions, dim=1).cpu().numpy())
            all_gts.append(masks.cpu().numpy())
            loop.set_postfix(loss=loss.item(), iou=iou)
    
    all_preds = np.concatenate(all_preds).flatten()
    all_gts = np.concatenate(all_gts).flatten()
    return total_loss / len(loader), total_iou / len(loader), all_gts, all_preds

def run_inference_on_image(image, model, transform):
    size = config.IMAGESIZE; stride = size // 2
    prediction_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    count_map = np.zeros_like(prediction_map, dtype=np.float32)

    for r in tqdm(range(0, image.shape[0] - size + 1, stride), desc="Inferring..."):
        for c in range(0, image.shape[1] - size + 1, stride):
            img_tile = image[r:r+size, c:c+size]
            with torch.no_grad():
                image_tensor = transform(image=img_tile)['image'].unsqueeze(0).to(DEVICE)
                prediction = model(image_tensor)
                predicted_mask = torch.argmax(prediction.squeeze(0), dim=0).cpu().numpy()
            prediction_map[r:r+size, c:c+size] += predicted_mask.astype(np.uint8)
            count_map[r:r+size, c:c+size] += 1
    
    return np.divide(prediction_map, count_map, where=count_map!=0).round().astype(np.uint8)

# =======================================================================================
# 6. VISUALIZATION AND REPORTING FUNCTIONS
# =======================================================================================
def normalize_for_display(img):
    p2, p98 = np.percentile(img[img > 0], (2, 98)) if (img > 0).any() else (0, 100)
    img_clipped = np.clip(img, p2, p98)
    return ((img_clipped - p2) / (p98 - p2) * 255).astype(np.uint8) if (p98 - p2) > 0 else img

def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Training History')
    ax1.plot(history['train_loss'], label='Train Loss'); ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Over Epochs'); ax1.legend()
    ax2.plot(history['train_iou'], label='Train IoU'); ax2.plot(history['val_iou'], label='Validation IoU')
    ax2.set_title('Mean IoU Over Epochs'); ax2.legend()
    plt.savefig(save_path, dpi=300); plt.close()
    print(f"Training history plot saved to: {save_path}")

def save_metrics_report(gts, preds, save_path_cm, save_path_report):
    cm = confusion_matrix(gts, preds); plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.title('Confusion Matrix'); plt.ylabel('Actual Class'); plt.xlabel('Predicted Class')
    plt.savefig(save_path_cm, dpi=300); plt.close()
    print(f"Confusion matrix saved to: {save_path_cm}")

    report = classification_report(gts, preds, target_names=config.CLASS_NAMES)
    print("\n--- Detailed Classification Report ---"); print(report)
    with open(save_path_report, 'w') as f: f.write(report)
    print(f"Classification report saved to: {save_path_report}")
    
def visualize_final_output(city, rgb_path, pred_mask_path, gt_mask_path=None):
    print("--- Creating Final Map Visualization ---")
    rgb_image = tifffile.imread(rgb_path)[:, :, 0:3]
    pred_mask = tifffile.imread(pred_mask_path)
    rgb_display = normalize_for_display(rgb_image)
    
    num_plots = 3 if gt_mask_path and os.path.exists(gt_mask_path) else 2
    fig, axs = plt.subplots(1, num_plots, figsize=(8 * num_plots, 8))
    
    cmap = ListedColormap(['#00000000', 'gray', 'red']) # 0=transparent
    
    axs[0].imshow(rgb_display); axs[0].set_title(f"Input RGB - {city.capitalize()}"); axs[0].axis('off')
    
    axs[1].imshow(rgb_display); axs[1].imshow(pred_mask, cmap=cmap, alpha=0.6)
    axs[1].set_title("Predicted Segmentation"); axs[1].axis('off')
    
    if num_plots == 3:
        gt_mask = tifffile.imread(gt_mask_path)
        axs[2].imshow(rgb_display); axs[2].imshow(gt_mask, cmap=cmap, alpha=0.6)
        axs[2].set_title("Ground Truth Segmentation"); axs[2].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, f"{city}_final_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0); plt.close()
    print(f"Final comparison visualization saved to: {save_path}")

# =======================================================================================
# 7. MAIN EXECUTION BLOCK (for training)
# =======================================================================================
if __name__ == '__main__':
    split_raster_function(config.TARGET_CITY, config.DATA_DIR, config.IMAGE_TILES_DIR, config.MASK_TILES_DIR)

    all_image_paths = sorted(glob.glob(os.path.join(config.IMAGE_TILES_DIR, "*.tif")))
    all_mask_paths = sorted(glob.glob(os.path.join(config.MASK_TILES_DIR, "*.tif")))

    if not all_image_paths:
        print("No training tiles found. Please run the `split_raster_function` first.")
    else:
        print(f"Found {len(all_image_paths)} tiles. Preparing for training...")
        img_paths_train, img_paths_val, mask_paths_train, mask_paths_val = train_test_split(
            all_image_paths, all_mask_paths, test_size=0.2, random_state=config.SEED
        )
        
        train_transform, val_transform = get_transforms()
        train_dataset = SegmentationDataset(img_paths_train, mask_paths_train, transform=train_transform)
        val_dataset = SegmentationDataset(img_paths_val, mask_paths_val, transform=val_transform)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

        model = UNet(n_channels=3, n_classes=config.CLASSES).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        best_iou = 0.0
        history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
        model_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_{config.TARGET_CITY}_best.pth")

        for epoch in range(config.EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
            train_loss, train_iou = train_one_epoch(train_loader, model, optimizer, loss_fn)
            val_loss, val_iou, _, _ = evaluate_model(val_loader, model, loss_fn)
            
            history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
            history['train_iou'].append(train_iou); history['val_iou'].append(val_iou)
            
            print(f"Epoch {epoch+1} -> Train Loss:{train_loss:.4f}, Train IoU:{train_iou:.4f} | Val Loss:{val_loss:.4f}, Val IoU:{val_iou:.4f}")
            
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), model_path)
                print(f"✅ Model Saved! New best validation IoU: {best_iou:.4f}")

        plot_history(history, os.path.join(config.RESULTS_DIR, f"{config.TARGET_CITY}_training_curves.png"))

        print("\n--- Generating Final Validation Metrics ---")
        model.load_state_dict(torch.load(model_path)) # Load best model
        _, _, final_gts, final_preds = evaluate_model(val_loader, model, loss_fn)
        save_metrics_report(
            final_gts, final_preds,
            os.path.join(config.RESULTS_DIR, f"{config.TARGET_CITY}_confusion_matrix.png"),
            os.path.join(config.RESULTS_DIR, f"{config.TARGET_CITY}_classification_report.txt")
        )
        
    print("\n\n--- Training Pipeline Finished ---")