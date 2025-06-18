# draw_importantpart_full_avg.py
# 每张图保存 + 原图热图 + crop热图 + 在线平均热图

import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from engine.core import YAMLConfig

# ======================= 配置路径 ======================= #
cfg_path = "/home/hanna/桌面/DEIM/configs/deim_dfine/deim_hgnetv2_s_rafdb.yml"
checkpoint_path = "/home/hanna/桌面/DEIM/outputs/deim_hgnetv2_s_coco/best_stg2.pth"
dataset_root = "/home/hanna/桌面/vHeat/dataset/rafdb_split_singel/train"
bbox_file = "/home/hanna/桌面/FER-YOLO-Mamba/train.txt"
save_dir = "feature_vis_outputs"
os.makedirs(save_dir, exist_ok=True)

# ======================= 模型加载 ======================= #
cfg = YAMLConfig(cfg_path)
model = cfg.model
model.eval()
state = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(state['model'], strict=False)

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# ======================= 人脸框加载 ======================= #
def load_bbox_mapping(txt_path):
    mapping = {}
    with open(txt_path, 'r') as f:
        for line in f:
            path_part, bbox_part = line.strip().split(' ', 1)
            bbox_values = list(map(int, bbox_part.strip().split(',')[:4]))
            mapping[os.path.abspath(path_part)] = bbox_values
    return mapping

def resize_bbox(bbox, original_size, target_size=(640, 640)):
    ow, oh = original_size
    tw, th = target_size
    return [int(bbox[0]/ow*tw), int(bbox[1]/oh*th), int(bbox[2]/ow*tw), int(bbox[3]/oh*th)]

bbox_dict = load_bbox_mapping(bbox_file)

# ======================= 可视化工具 ======================= #
def visualize_feat_set(feat_list, title_prefix, fig, row_idx, max_cols=3, bbox=None):
    for i, feat in enumerate(feat_list):
        feat_up = F.interpolate(feat, size=(640, 640), mode='bilinear', align_corners=False)
        fmap = feat_up[0].mean(dim=0)
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)
        ax = fig.add_subplot(7, max_cols, row_idx * max_cols + i + 1)
        ax.imshow(fmap.cpu(), cmap='jet')
        ax.set_title(f"{title_prefix}[{i}]")
        ax.axis('off')

        if bbox:
            x1, y1, x2, y2 = bbox
            cropped = feat_up[0, :, y1:y2, x1:x2]
            if cropped.numel() > 0:
                fmap_crop = cropped.mean(dim=0)
                fmap_crop = (fmap_crop - fmap_crop.min()) / (fmap_crop.max() - fmap_crop.min() + 1e-5)
                ax_crop = fig.add_subplot(7, max_cols, (row_idx + 3) * max_cols + i + 1)
                ax_crop.imshow(fmap_crop.cpu(), cmap='jet')
                ax_crop.set_title(f"{title_prefix}[{i}]_crop")
                ax_crop.axis('off')

def update_online_avg(feat_list, bbox, avg_dict, feat_type):
    for i, feat in enumerate(feat_list):
        feat_up = F.interpolate(feat, size=(640, 640), mode='bilinear', align_corners=False)
        x1, y1, x2, y2 = bbox
        if x2 > x1 and y2 > y1:
            crop = feat_up[0, :, y1:y2, x1:x2]
            if crop.numel() == 0:
                continue
            crop_resized = F.interpolate(crop.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)[0].detach().cpu()
            if avg_dict[feat_type][i]['sum'] is None:
                avg_dict[feat_type][i]['sum'] = crop_resized
            else:
                avg_dict[feat_type][i]['sum'] += crop_resized
            avg_dict[feat_type][i]['count'] += 1

def draw_avg_heatmaps(avg_dict, save_dir, category_name):
    os.makedirs(save_dir, exist_ok=True)
    heatmaps = []
    for feat_type in avg_dict:
        for i, record in enumerate(avg_dict[feat_type]):
            if record['count'] == 0:
                continue
            avg_feat = record['sum'] / record['count']
            fmap = avg_feat.mean(dim=0)
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)
            heatmaps.append((f"{feat_type}[{i}]", fmap))

    cols = 3
    rows = (len(heatmaps) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    for idx, (title, fmap) in enumerate(heatmaps):
        r, c = divmod(idx, cols)
        ax = axs[r, c] if rows > 1 else axs[c]
        ax.imshow(fmap.numpy(), cmap='jet')
        ax.set_title(title)
        ax.axis('off')
    for j in range(len(heatmaps), rows * cols):
        r, c = divmod(j, cols)
        ax = axs[r, c] if rows > 1 else axs[c]
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{category_name}_avg_all.png"))
    plt.close()

# ======================= 主函数 ======================= #
def process_one_class(category_name, max_images=100):
    path = os.path.join(dataset_root, category_name)
    files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png'))])[:max_images]
    if not files:
        return
    save_path = os.path.join(save_dir, category_name)
    os.makedirs(save_path, exist_ok=True)

    avg_dict = {k: [{'sum': None, 'count': 0} for _ in range(3)] for k in ['proj_feats', 'inner_outs', 'outs']}

    for idx, fname in enumerate(files):
        img_path = os.path.join(path, fname)
        abs_path = os.path.abspath(img_path)
        if abs_path not in bbox_dict:
            continue
        image = Image.open(img_path).convert("RGB")
        bbox = resize_bbox(bbox_dict[abs_path], image.size)
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            feats = model.backbone(tensor)
            proj_feats = [model.encoder.input_proj[i](f) for i, f in enumerate(feats)]

            if model.encoder.num_encoder_layers > 0:
                for i, idx_enc in enumerate(model.encoder.use_encoder_idx):
                    h, w = proj_feats[idx_enc].shape[2:]
                    src = proj_feats[idx_enc].flatten(2).permute(0, 2, 1)
                    pos = model.encoder.build_2d_sincos_position_embedding(w, h, model.encoder.hidden_dim, model.encoder.pe_temperature)
                    memory = model.encoder.encoder[i](src, pos_embed=pos.to(src.device))
                    proj_feats[idx_enc] = memory.permute(0, 2, 1).reshape(-1, model.encoder.hidden_dim, h, w)

            inner_outs = [proj_feats[-1]]
            for i in range(len(model.encoder.in_channels) - 1, 0, -1):
                high = model.encoder.lateral_convs[len(model.encoder.in_channels) - 1 - i](inner_outs[0])
                up = F.interpolate(high, scale_factor=2., mode='nearest')
                low = proj_feats[i - 1]
                inner = model.encoder.fpn_blocks[len(model.encoder.in_channels) - 1 - i](torch.cat([up, low], 1))
                inner_outs.insert(0, inner)

            outs = [inner_outs[0]]
            for i in range(len(model.encoder.in_channels) - 1):
                down = model.encoder.downsample_convs[i](outs[-1])
                out = model.encoder.pan_blocks[i](torch.cat([down, inner_outs[i + 1]], 1))
                outs.append(out)

        fig = plt.figure(figsize=(15, 28))
        ax0 = fig.add_subplot(7, 3, 1)
        ax0.imshow(image)
        ax0.set_title("Original Image")
        ax0.axis('off')

        visualize_feat_set(proj_feats, "proj_feats", fig, 1, bbox=bbox)
        visualize_feat_set(inner_outs, "inner_outs", fig, 2, bbox=bbox)
        visualize_feat_set(outs, "outs", fig, 3, bbox=bbox)

        update_online_avg(proj_feats, bbox, avg_dict, 'proj_feats')
        update_online_avg(inner_outs, bbox, avg_dict, 'inner_outs')
        update_online_avg(outs, bbox, avg_dict, 'outs')

        save_img = os.path.join(save_path, f"{os.path.splitext(fname)[0]}.png")
        plt.tight_layout()
        plt.savefig(save_img)
        plt.close()
        print(f"Saved: {save_img}")

    draw_avg_heatmaps(avg_dict, save_path, category_name + "_avg")

if __name__ == '__main__':
    process_one_class("Surprise")