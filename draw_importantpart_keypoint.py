import os
import gc
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from engine.core import YAMLConfig
from scipy.interpolate import splprep, splev
import numpy as np
import matplotlib.cm as cm

# ======================= 配置路径 =======================
cfg_path = "/home/hanna/桌面/DEIM/configs/deim_dfine/deim_hgnetv2_s_rafdb.yml"
checkpoint_path = "/home/hanna/桌面/DEIM/outputs/deim_hgnetv2_s_coco/best_stg2.pth"
dataset_root = "/home/hanna/桌面/vHeat/dataset/rafdb_split_singel/train"
bbox_file = "/home/hanna/桌面/FER-YOLO-Mamba/train.txt"
kpt_root = "/home/hanna/桌面/rafdb_basic./Annotation/auto"
save_dir = "feature_vis_outputs"
os.makedirs(save_dir, exist_ok=True)

# ======================= 加载模型 =======================
cfg = YAMLConfig(cfg_path)
model = cfg.model
model.eval()
state = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(state['model'], strict=False)

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# ======================= 加载BBOX =======================
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

# ======================= 绘制关键点曲线 =======================
def draw_keypoints_overlay(image_pil, keypoints):
    draw = ImageDraw.Draw(image_pil)
    regions = {
        "left_eyebrow": list(range(0, 3)),
        "right_eyebrow": list(range(3, 6)),
        "nose": list(range(6, 8)),
        "nose_blow": list(range(8, 13)),
        "left_eye": list(range(13, 19)),
        "right_eye": list(range(19, 25)),
        "mouth": list(range(25, 37))
    }
    for name, indices in regions.items():
        pts = [keypoints[i] for i in indices]
        if len(pts) < 2:
            continue
        points_np = list(zip(*pts))
        k = min(3, len(pts) - 1)
        try:
            tck, u = splprep(points_np, s=0, per=(name in ["left_eye", "right_eye", "mouth"]), k=k)
            u_fine = np.linspace(0, 1, 100)
            x_fine, y_fine = splev(u_fine, tck)
            for i in range(len(x_fine) - 1):
                draw.line([(x_fine[i], y_fine[i]), (x_fine[i+1], y_fine[i+1])], fill=(0, 255, 0), width=1)
        except:
            pass
    return image_pil

# ======================= 处理类别 =======================
def process_one_class(category_name, max_images=3):
    path = os.path.join(dataset_root, category_name)
    files = sorted([f for f in os.listdir(path) if f.lower().endswith('.jpg')])[:max_images]
    save_path = os.path.join(save_dir, category_name,'keypoint')
    os.makedirs(save_path, exist_ok=True)

    for fname in files:
        img_path = os.path.join(path, fname)
        abs_path = os.path.abspath(img_path)
        if abs_path not in bbox_dict:
            continue

        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        bbox = resize_bbox(bbox_dict[abs_path], image.size)

        with torch.no_grad():
            feats = model.backbone(tensor)
            proj_feats = [model.encoder.input_proj[i](f) for i, f in enumerate(feats)]

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

        feat_up = F.interpolate(outs[0], size=(640, 640), mode='bilinear', align_corners=False)
        fmap = feat_up[0].mean(dim=0).cpu()
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)
        cmap = cm.jet(fmap.numpy())[:, :, :3]  # Remove alpha
        heatmap_np = (cmap * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_np)

        kpt_path = os.path.join(kpt_root, f"{os.path.splitext(fname)[0]}_auto_attri.txt")
        keypoints = []
        try:
            with open(kpt_path, 'r') as f:
                for line in f:
                    x, y = map(float, line.strip().split())
                    keypoints.append((int(x / image.width * 640), int(y / image.height * 640)))
        except:
            keypoints = []

        if len(keypoints) >= 37:
            heatmap_img = draw_keypoints_overlay(heatmap_img, keypoints)

        save_img = os.path.join(save_path, f"{os.path.splitext(fname)[0]}_outs0_kpt.jpg")
        heatmap_img.save(save_img)
        print(f"✅ Saved: {save_img}")

        del feats, proj_feats, tensor, fmap, heatmap_np
        gc.collect()

if __name__ == '__main__':
    process_one_class("Neutral")