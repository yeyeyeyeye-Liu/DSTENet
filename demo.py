import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import roi_align

# =====================================================
# DropPath
# =====================================================
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()


# =====================================================
# Deformable RoI Pooling
# =====================================================
class DeformableRoIPooling(nn.Module):
    def __init__(self, in_channels, output_size, spatial_scale=1.0, gamma=0.1):
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
        pooled = roi_align(
            features, rois,
            self.output_size,
            spatial_scale=self.spatial_scale,
            aligned=True
        )
        offsets = self.offset_net(pooled)
        return self.deformable_pooling(pooled, offsets, rois)

    def deformable_pooling(self, pooled, offsets, rois):
        N, C, H, W = pooled.shape
        device = pooled.device
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        base_grid = torch.stack((x, y), dim=-1).unsqueeze(0).repeat(N, 1, 1, 1)
        roi_sizes = (rois[:, 3:5] - rois[:, 1:3]).unsqueeze(-1).unsqueeze(-1)
        roi_sizes = roi_sizes.permute(0, 2, 3, 1)
        offset_grid = offsets.permute(0, 2, 3, 1) * self.gamma * roi_sizes
        grid = base_grid + offset_grid
        return F.grid_sample(pooled, grid, align_corners=True)


# =====================================================
# PR2D Block
# =====================================================
class PR2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, deformable=True):
        super().__init__()
        self.deformable = deformable

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.drop_path = DropPath(0.1)

        if deformable:
            self.pool = DeformableRoIPooling(in_channels, (8, 8))
            self.pool_adjust = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.pool = nn.AvgPool2d(stride, stride)
            self.pool_adjust = nn.Identity()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        conv_out = self.conv(x)

        if self.deformable:
            rois = self.full_roi(x)
            pool_out = self.pool(x, rois)
            pool_out = self.pool_adjust(pool_out)
        else:
            pool_out = self.pool(x)

        pool_out = F.adaptive_avg_pool2d(pool_out, conv_out.shape[-2:])
        shortcut = F.adaptive_avg_pool2d(self.shortcut(x), conv_out.shape[-2:])
        return F.relu(conv_out + shortcut + self.drop_path(pool_out), inplace=True)

    def full_roi(self, x):
        rois = []
        for i in range(x.size(0)):
            h, w = x.shape[2:]
            rois.append([i, 0, 0, w - 1, h - 1])
        return torch.tensor(rois, device=x.device, dtype=torch.float32)


# =====================================================
# ConvNeXt Block
# =====================================================
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(0.1)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = F.gelu(self.pwconv1(x))
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


# =====================================================
# Transformer Bridge
# =====================================================
class SpatialTransformer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        out = self.encoder(x_flat)
        out = out.transpose(1, 2).view(B, C, H, W)
        return x + self.gamma * out


# =====================================================
# DSTENet
# =====================================================
class DSTENet(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(8, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.pr1 = PR2D(32, 64)
        self.pr2 = PR2D(64, 128)
        self.convnext = ConvNeXtBlock(128)
        self.reduce = nn.Conv2d(128, feature_dim, 1)
        self.transformer = SpatialTransformer(feature_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 8)
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.pr1(x)
        x = self.pr2(x)
        x = self.convnext(x)
        x = self.reduce(x)
        x = self.transformer(x)
        x = self.refine(x)
        return self.head(x)


# =====================================================
# Demo
# =====================================================
def main():
    print("=== DSTENet Demo Run ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data = np.load("./Data_instance/Data_instance.npy")
    print("Original dtype:", data.dtype)
    print("Original shape:", data.shape)

    data = data.astype(np.float32)

    if data.shape == (8, 1280, 1280):
        data = torch.from_numpy(data).unsqueeze(0)
        data = F.interpolate(data, size=(224, 224), mode="bilinear", align_corners=False)
    else:
        raise ValueError("Expected shape (8, 1280, 1280)")

    data = data.to(device)

    model = DSTENet().to(device)
    model.eval()

    with torch.no_grad():
        out = model(data)

    print("Output logits:", out.cpu().numpy())
    print("Predicted class:", torch.argmax(out, dim=1).item())
    print("=== Demo finished successfully ===")


if __name__ == "__main__":
    main()