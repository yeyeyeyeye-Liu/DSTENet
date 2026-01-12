import random
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from torchvision.transforms import transforms
from torchvision.ops import roi_align
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# Custom dataset for loading 8-channel hyperspectral/multispectral data
class SimpleDataset(Dataset):
    def __init__(self, data_file, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform_d = transform

    def __getitem__(self, i):
        # Load data from numpy file
        data_path = Path(self.meta['data_path'][i])
        data = np.load(data_path, allow_pickle=True)
        
        # Convert to tensor
        data = torch.from_numpy(data).float()
        
        # Reshape/resize if necessary
        # Resize large input patches (1280×1280) to 224×224 to match network input resolution
        if data.shape == (8, 1280, 1280):
            data = data.unsqueeze(0)
            data = F.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        elif data.shape != (8, 224, 224):
            raise ValueError(f"Unexpected shape {data.shape} at index {i}")
            
        data = self.transform_d(data)
        label = int(self.meta['data_labels'][i])
        return data, label

    def __len__(self):
        return len(self.meta['data_labels'])

# Parameter configuration
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--class_num', type=int, default=8)  # 8 classes
option = parser.parse_known_args()[0]

# Data preprocessing
transform = transforms.Compose([
    transforms.Normalize(mean=[0.5] * 8, std=[0.5] * 8)
])

# Create datasets and dataloaders
train_dataset = SimpleDataset('./train4.json', transform)
train_dataloader = DataLoader(train_dataset, batch_size=option.batch_size, 
                             shuffle=True, num_workers=option.num_workers)

test_dataset = SimpleDataset('./test4.json', transform)
test_dataloader = DataLoader(test_dataset, batch_size=option.batch_size, 
                            shuffle=False, num_workers=option.num_workers)

# ===== DropPath Module =====
class DropPath(nn.Module):
    """Stochastic depth regularization (drop path) for residual networks"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()

# ===== Deformable RoI Pooling =====
class DeformableRoIPooling(nn.Module):
    """Deformable Region of Interest Pooling with learnable offsets"""
    def __init__(self, in_channels, output_size, spatial_scale, gamma=0.1):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.gamma = gamma
        self.offset_net = nn.Sequential(
            nn.Conv2d(in_channels, max(64, in_channels // 4), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(64, in_channels // 4), max(32, in_channels // 8), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(32, in_channels // 8), 2, 1)
        )

    def forward(self, features, rois):
        # Regular RoI pooling
        pooled = roi_align(features, rois, self.output_size, spatial_scale=self.spatial_scale, aligned=True)
        # Learn offset adjustment
        offsets = self.offset_net(pooled)
        return self.deformable_pooling(pooled, offsets, rois)

    def deformable_pooling(self, pooled, offsets, rois):
        N, C, H, W = pooled.shape
        device = pooled.device
        # Create base grid
        y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device),
                              torch.linspace(-1, 1, W, device=device), indexing='ij')
        base_grid = torch.stack((x, y), dim=-1).unsqueeze(0).repeat(N, 1, 1, 1)
        # Calculate offset grid
        roi_sizes = (rois[:, 3:5] - rois[:, 1:3]).unsqueeze(-1).unsqueeze(-1).permute(0, 2, 3, 1)
        offset_grid = offsets.permute(0, 2, 3, 1) * self.gamma * roi_sizes
        deformed_grid = base_grid + offset_grid
        return F.grid_sample(pooled, deformed_grid, align_corners=True)

# ===== PR2D Module (with DropPath) =====
class PR2D(nn.Module):
    """Pyramidal Residual 2D block with deformable pooling option"""
    def __init__(self, in_channels, out_channels, stride=2, deformable=False, drop_prob=0.1):
        super().__init__()
        self.deformable = deformable
        # Main convolution path
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.drop_path = DropPath(drop_prob)
        # Pooling path
        if deformable:
            self.pool = DeformableRoIPooling(in_channels, (8, 8), 1.0)
        else:
            self.pool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        # Channel adjustment for pooling path
        self.pool_channel_adjust = nn.Sequential()
        if in_channels != out_channels and deformable:
            self.pool_channel_adjust = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        conv_out = self.conv(x)
        # Pooling path processing
        if self.deformable:
            rois = self.create_full_roi(x)
            pool_out = self.pool(x, rois)
            pool_out = self.pool_channel_adjust(pool_out)
        else:
            pool_out = self.pool(x)
            if pool_out.shape[1] != conv_out.shape[1]:
                pool_out = nn.Conv2d(pool_out.shape[1], conv_out.shape[1], 1)(pool_out)
        # Adaptive pooling to match dimensions
        pool_out = F.adaptive_avg_pool2d(pool_out, conv_out.shape[-2:])
        shortcut = self.shortcut(x)
        shortcut = F.adaptive_avg_pool2d(shortcut, conv_out.shape[-2:])
        if shortcut.shape[1] != conv_out.shape[1]:
            shortcut = nn.Conv2d(shortcut.shape[1], conv_out.shape[1], 1)(shortcut)
        return F.relu(conv_out + shortcut + self.drop_path(pool_out), inplace=True)

    def create_full_roi(self, x):
        """Create RoIs covering entire feature maps"""
        rois = []
        for i in range(x.size(0)):
            h, w = x.shape[2:]
            rois.append([i, 0, 0, w - 1, h - 1])
        return torch.tensor(rois, dtype=torch.float, device=x.device)

# ===== ConvNeXt Block =====
class ConvNeXtBlock(nn.Module):
    """Modern ConvNeXt-style block with depthwise convolution"""
    def __init__(self, dim, drop_prob=0.1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_prob)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)

# ===== Transformer Bridge (Weighted Residual) =====
class SpatialTransformer(nn.Module):
    """Transformer module for spatial feature enhancement"""
    def __init__(self, dim, num_heads=4, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.gamma = nn.Parameter(torch.tensor(0.1))  # Learnable weighting factor

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        trans_out = self.transformer(x_flat)
        trans_out = trans_out.transpose(1, 2).view(B, C, H, W)
        return x + self.gamma * trans_out  # Weighted residual connection

# ===== Main Backbone Network =====
class DSTENet(nn.Module):
    """Main network architecture for 8-class classification"""
    def __init__(self, feature_dim=256, use_deformable=True):
        super().__init__()
        # Initial preprocessing
        self.pre_layer = nn.Sequential(
            nn.Conv2d(8, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # PR2D blocks
        self.pr2d1 = PR2D(32, 64, stride=2, deformable=use_deformable)
        self.pr2d2 = PR2D(64, 128, stride=2, deformable=use_deformable)
        # ConvNeXt enhancement
        self.convnext = ConvNeXtBlock(128)
        self.convnext_down = nn.Conv2d(128, feature_dim, 1)
        # Transformer for spatial relationships
        self.transformer = SpatialTransformer(feature_dim, num_heads=4, num_layers=1)
        # Refinement convolution
        self.refinement = nn.Sequential(
            nn.Conv2d(feature_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 8)
        )

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.pr2d1(x)
        x = self.pr2d2(x)
        x = self.convnext(x)
        x = self.convnext_down(x)
        x = self.transformer(x)
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.refinement(x)
        return self.classifier(x)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DSTENet(feature_dim=256, use_deformable=True).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=option.lr, momentum=0.9, weight_decay=1e-4)
ce_loss = nn.CrossEntropyLoss().to(device)

# Training logs
log_train_loss, log_train_acc = [], []
log_test_loss, log_test_acc, log_test_f1 = [], [], []
log_test_precision, log_test_recall, log_test_aa, log_test_kappa = [], [], [], []
log_best_epoch, log_best_acc = 0, 0
log_best_pred, log_test_real = None, None

# Import metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score
)

# ============================
# Main Training Loop
# ============================
for epoch in range(1, option.max_epoch + 1):
    # ===== Training Phase =====
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for i, (data, label) in enumerate(train_dataloader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = ce_loss(output, label)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        train_correct += predicted.eq(label).sum().item()
        train_total += label.size(0)
        train_loss += loss.item()

    train_acc = 100. * train_correct / train_total
    avg_train_loss = train_loss / (i + 1)
    print(f'[Epoch:{epoch:3d}] AvgLoss: {avg_train_loss:.5f} | Acc: {train_acc:.3f}%')
    log_train_loss.append(round(avg_train_loss, 3))
    log_train_acc.append(round(train_acc, 3))

    # ===== Testing Phase =====
    model.eval()
    test_correct, test_total, loss_test = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, label in test_dataloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = ce_loss(output, label)
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            test_correct += predicted.eq(label).sum().item()
            test_total += label.size(0)
            loss_test += loss.item()

    # ===== Calculate Metrics =====
    loss_test /= len(test_dataloader)
    test_acc = 100. * test_correct / test_total
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    aa = recall  # Average Accuracy equals macro recall
    kappa = cohen_kappa_score(all_labels, all_preds)

    # ===== Display Metrics =====
    print(f'\n[Test] Metrics at Epoch {epoch}:')
    print(f'  Test Acc (OA):   {test_acc:.3f}%')
    print(f'  Test Loss:       {loss_test:.5f}')
    print(f'  Macro F1 Score:  {f1:.4f}')
    print(f'  Macro Precision: {precision:.4f}')
    print(f'  Macro Recall (AA): {recall:.4f}')
    print(f'  Kappa Score:     {kappa:.4f}')
    print('  Classification Report:\n', classification_report(all_labels, all_preds, digits=4))
    print('  Confusion Matrix:\n', confusion_matrix(all_labels, all_preds))

    # ===== Log Metrics =====
    log_test_acc.append(round(test_acc, 3))
    log_test_loss.append(round(loss_test, 5))
    log_test_f1.append(round(f1, 4))
    log_test_precision.append(round(precision, 4))
    log_test_recall.append(round(recall, 4))
    log_test_aa.append(round(aa, 4))
    log_test_kappa.append(round(kappa, 4))

    # ===== Save Best Model =====
    if test_acc >= log_best_acc and epoch > 10:
        log_best_epoch = epoch
        log_best_acc = test_acc
        log_best_pred = all_preds
        log_test_real = all_labels
        torch.save(model.state_dict(), 'DSTENet_best.pth')
        print(f'\nSaved best model at epoch {epoch} with test acc {test_acc:.2f}%')

# ===== Training Complete =====
print(f'\nTraining complete. Best model at epoch {log_best_epoch} with test acc {log_best_acc:.2f}%. '
      f'Saved as DSTENet_best.pth')
