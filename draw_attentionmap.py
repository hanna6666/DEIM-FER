# import os
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from torchvision.transforms.functional import to_pil_image
# from PIL import Image
# from engine.core import YAMLConfig

# # ========== 配置路径 ==========
# cfg_path = "/home/hanna/桌面/DEIM/configs/deim_dfine/deim_hgnetv2_s_rafdb.yml"
# checkpoint_path = "/home/hanna/桌面/DEIM/outputs/deim_hgnetv2_s_coco/best_stg2.pth"
# image_path = "/home/hanna/桌面/vHeat/dataset/rafdb_split_singel/train/Anger/train_06522.jpg"

# # ========== 模型加载 ==========
# cfg = YAMLConfig(cfg_path)
# model = cfg.model
# model.eval()
# state = torch.load(checkpoint_path, map_location='cpu')
# model.load_state_dict(state['model'], strict=False)

# # ========== 图像预处理 ==========
# def load_image(image_path, size=640):
#     tf = transforms.Compose([
#         transforms.Resize((size, size)),
#         transforms.ToTensor(),
#     ])
#     image = Image.open(image_path).convert("RGB")
#     return tf(image)

# image_tensor = load_image(image_path, size=640)
# input_tensor = image_tensor.unsqueeze(0)

# # ========== 注册 hook 拿到 Q/K 输入 ==========
# qk_cache = {}

# def hook_raw_qk(module, input, output):
#     qk_cache['q'] = input[0].detach()  # 输入 query: [B, N, C]
#     qk_cache['k'] = input[1].detach()  # 输入 key: [B, N, C]

# target_layer = model.encoder.encoder[-1].layers[-1].self_attn
# hook_handle = target_layer.register_forward_hook(hook_raw_qk)

# # ========== 前向传播 ==========
# with torch.no_grad():
#     _ = model(input_tensor)

# hook_handle.remove()

# # ========== 准备 QK linear 权重 ==========
# C = qk_cache['q'].shape[-1]
# H = target_layer.num_heads
# dim_head = C // H
# W_qkv = target_layer.in_proj_weight  # [3C, C]
# b_qkv = target_layer.in_proj_bias    # [3C]

# # Q projection
# q_w = W_qkv[:C, :]
# q_b = b_qkv[:C]
# q_proj = F.linear(qk_cache['q'], q_w, q_b)

# # K projection
# k_w = W_qkv[C:2*C, :]
# k_b = b_qkv[C:2*C]
# k_proj = F.linear(qk_cache['k'], k_w, k_b)

# # ========== 变成多头形式 ==========
# B, N, _ = q_proj.shape
# q = q_proj.view(B, N, H, dim_head).permute(0, 2, 1, 3)  # [B, H, N, d]
# k = k_proj.view(B, N, H, dim_head).permute(0, 2, 1, 3)  # [B, H, N, d]

# # ========== QKᵀ 不加 softmax ==========
# scaling = dim_head ** -0.5
# q = q * scaling
# raw_scores = torch.matmul(q, k.transpose(-2, -1))  # [B, H, N, N]
# print("raw QKᵀ shape:", raw_scores.shape)

# # ========== 可视化某个 token 的每个 head ==========
# def visualize_token_raw_attention(raw_scores, image_tensor, token_idx=85):
#     B, H, N, _ = raw_scores.shape
#     spatial_size = int(N ** 0.5)
#     fig, axes = plt.subplots(1, H + 1, figsize=(3 * (H + 1), 4))

#     axes[0].imshow(to_pil_image(image_tensor.detach().cpu()))
#     axes[0].set_title("Original")
#     axes[0].axis("off")

#     for h in range(H):
#         attn_1d = raw_scores[0, h, token_idx]  # [N]
#         attn_2d = attn_1d.view(spatial_size, spatial_size)
#         attn_2d = F.interpolate(attn_2d[None, None], size=image_tensor.shape[1:], mode='bilinear')[0, 0]
#         attn_2d = attn_2d - attn_2d.min()
#         attn_2d = attn_2d / (attn_2d.max() + 1e-6)

#         axes[h + 1].imshow(to_pil_image(image_tensor.detach().cpu()))
#         axes[h + 1].imshow(attn_2d.detach().cpu(), cmap='jet', alpha=0.5)
#         axes[h + 1].set_title(f"Head {h}")
#         axes[h + 1].axis("off")

#     plt.tight_layout()
#     plt.show()

# def visualize_token_raw_attention_avg(raw_scores, image_tensor, token_idx=0):
#     """
#     可视化所有 head 的平均注意力图（不经过 softmax）
#     """
#     B, H, N, _ = raw_scores.shape
#     spatial_size = int(N ** 0.5)

#     # 取出所有 head 的 attention，然后平均
#     attn_1d_all = raw_scores[0, :, token_idx, :]  # [H, N]
#     attn_1d_avg = attn_1d_all.mean(dim=0)  # [N]

#     # reshape 成 2D 空间结构
#     attn_2d = attn_1d_avg.view(spatial_size, spatial_size)
#     attn_2d = F.interpolate(attn_2d[None, None], size=image_tensor.shape[1:], mode='bilinear')[0, 0]

#     # 归一化
#     attn_2d = attn_2d - attn_2d.min()
#     attn_2d = attn_2d / (attn_2d.max() + 1e-6)

#     # 显示图像
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.imshow(to_pil_image(image_tensor.detach().cpu()))
#     ax.imshow(attn_2d.detach().cpu(), cmap='jet', alpha=0.5)
#     ax.set_title(f"Token {token_idx} | Avg of {H} Heads")
#     ax.axis("off")
#     plt.tight_layout()
#     plt.show()

# # ========== 改这里的 index 看其他 token ==========
# visualize_token_raw_attention_avg(raw_scores, image_tensor, token_idx=85)


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from engine.core import YAMLConfig  # 确保你本地环境正确引入

# ==== 路径设置 ====
cfg_path = "/home/hanna/桌面/DEIM/configs/deim_dfine/deim_hgnetv2_s_rafdb.yml"
checkpoint_path = "/home/hanna/桌面/DEIM/outputs/deim_hgnetv2_s_coco/best_stg2.pth"
image_path = "/home/hanna/桌面/vHeat/dataset/rafdb_split_singel/train/Anger/train_06522.jpg"

# ==== 加载模型 ====
cfg = YAMLConfig(cfg_path)
model = cfg.model
model.eval()
state = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(state['model'], strict=False)

# ==== 加载图片 ====
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image)
input_tensor = image_tensor.unsqueeze(0)

# ==== 注册 Hook 以捕获 encoder attention ====
attn_maps = {}
def hook_fn(module, input, output):
    attn_maps["encoder_attn"] = output[1].detach()

# 取 encoder 最后一层 self-attn
target_layer = model.encoder.encoder[-1].layers[-1].self_attn
target_layer.register_forward_hook(hook_fn)

# ==== 推理 ====
with torch.no_grad():
    _ = model(input_tensor)

# ==== 提取注意力 ====
attn = attn_maps["encoder_attn"]  # [1, heads, Q, K]
attn = attn[0]  # [heads, Q, K]
num_heads, Q, K = attn.shape
token_idx = 85  # 选择一个 Query token
spatial_size = int(K ** 0.5)

# [heads, H_, W_]
attn_per_head = attn[:, token_idx, :].view(num_heads, spatial_size, spatial_size)

# 上采样并取平均
upsampled = F.interpolate(attn_per_head.unsqueeze(1), size=image_tensor.shape[1:], mode='bilinear', align_corners=False).squeeze(1)
avg_heatmap = upsampled.mean(dim=0)
avg_heatmap = avg_heatmap / avg_heatmap.max()

# ==== 可视化 ====
plt.figure(figsize=(8, 6))
plt.imshow(to_pil_image(image_tensor))
plt.imshow(avg_heatmap.cpu(), cmap='jet', alpha=0.5)
plt.title(f'Encoder Attention (Token {token_idx})')
plt.axis('off')
plt.show()