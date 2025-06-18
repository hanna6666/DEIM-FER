import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from engine.core import YAMLConfig
import numpy as np

# ==== 配置路径 ====
cfg_path = "/home/hanna/桌面/DEIM/configs/deim_dfine/deim_hgnetv2_s_rafdb.yml"
checkpoint_path = "/home/hanna/桌面/DEIM/outputs/deim_hgnetv2_s_coco/best_stg2.pth"
image_path = "/home/hanna/桌面/vHeat/dataset/rafdb_split_singel/train/Happiness/train_00338.jpg"

# ==== 加载模型 ====
cfg = YAMLConfig(cfg_path)
model = cfg.model
model.eval()
state = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(state['model'], strict=False)

# ==== 图像预处理 ====
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image)
input_tensor = image_tensor.unsqueeze(0)

# ==== 存储特征图 ====
proj_feats_result = []
inner_outs_result = []
outs_result = []

def forward_and_capture(model, x):
    feats = model.backbone(x)
    proj_feats = [model.encoder.input_proj[i](feat) for i, feat in enumerate(feats)]
    proj_feats_result.extend(proj_feats)

    if model.encoder.num_encoder_layers > 0:
        for i, enc_ind in enumerate(model.encoder.use_encoder_idx):
            h, w = proj_feats[enc_ind].shape[2:]
            src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
            pos_embed = model.encoder.build_2d_sincos_position_embedding(
                w, h, model.encoder.hidden_dim, model.encoder.pe_temperature).to(src_flatten.device)
            memory = model.encoder.encoder[i](src_flatten, pos_embed=pos_embed)
            proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, model.encoder.hidden_dim, h, w).contiguous()

    inner_outs = [proj_feats[-1]]
    for idx in range(len(model.encoder.in_channels) - 1, 0, -1):
        feat_heigh = inner_outs[0]
        feat_low = proj_feats[idx - 1]
        feat_heigh = model.encoder.lateral_convs[len(model.encoder.in_channels) - 1 - idx](feat_heigh)
        inner_outs[0] = feat_heigh
        upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
        inner_out = model.encoder.fpn_blocks[len(model.encoder.in_channels) - 1 - idx](torch.cat([upsample_feat, feat_low], dim=1))
        inner_outs.insert(0, inner_out)
    inner_outs_result.extend(inner_outs)

    outs = [inner_outs[0]]
    for idx in range(len(model.encoder.in_channels) - 1):
        feat_low = outs[-1]
        feat_height = inner_outs[idx + 1]
        downsample_feat = model.encoder.downsample_convs[idx](feat_low)
        out = model.encoder.pan_blocks[idx](torch.cat([downsample_feat, feat_height], dim=1))
        outs.append(out)
    outs_result.extend(outs)

with torch.no_grad():
    _ = forward_and_capture(model, input_tensor)

# ==== 可视化函数 ====
def show_feature_maps_grid(feat_list, title_prefix, max_cols=4):
    num_feats = len(feat_list)
    cols = min(num_feats, max_cols)
    rows = (num_feats + cols - 1) // cols  # 向上取整

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, feat in enumerate(feat_list):
        fmap = feat[0].mean(dim=0)  # 平均通道
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)
        axes[i].imshow(fmap.cpu(), cmap='jet')
        axes[i].set_title(f"{title_prefix}[{i}] - {feat.shape}")
        axes[i].axis('off')

    # 隐藏多余的 subplot 坐标轴
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# ==== 显示热力图 ====
show_feature_maps_grid(proj_feats_result, "proj_feats")
show_feature_maps_grid(inner_outs_result, "inner_outs")
show_feature_maps_grid(outs_result, "outs")