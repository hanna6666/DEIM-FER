import cv2
import numpy as np
from scipy.interpolate import splprep, splev

# ======= 路径设置 =======
keypoint_file = "/home/hanna/桌面/rafdb_basic./Annotation/auto/train_09209_auto_attri.txt"
image_path = "/home/hanna/桌面/rafdb_basic./Image/original/train_09209.jpg"
output_path = "/home/hanna/桌面/DEIM/train_09209_kpt_mask_weighted_spline.jpg"

# ======= 参数设置 =======
sigma = 10                # 高斯核扩散范围
spline_num_points = 30    # 拟合点数量
normalize_output = True   # 是否最终归一化热图值到 0~1

# ======= 加载图像与关键点 =======
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"❌ 图像文件未找到: {image_path}")
H, W = image.shape[:2]

keypoints = []
with open(keypoint_file, 'r') as f:
    for line in f:
        x, y = map(float, line.strip().split())
        keypoints.append((int(x), int(y)))

if len(keypoints) != 37:
    raise ValueError(f"❌ 关键点数量应为 37，当前为 {len(keypoints)}")

# ======= 区域定义（附带权重）=======
regions = {
    "left_eyebrow":  {"indices": list(range(0, 3)),   "weight": 0.1},
    "right_eyebrow": {"indices": list(range(3, 6)),   "weight": 0.1},
    # "nose":          {"indices": list(range(6, 8)),   "weight": 1.0},
    "nose_blow":     {"indices": list(range(7, 13)),  "weight": 0.4},
    "left_eye":      {"indices": list(range(13, 19)), "weight": 1.0},
    "right_eye":     {"indices": list(range(19, 25)), "weight": 1.0},
    "mouth":         {"indices": list(range(25, 37)), "weight": 1.2}
}

# ======= 工具函数 =======
def draw_gaussian(heatmap, center, sigma):
    x, y = center
    tmp_size = sigma * 3
    ul = [int(x - tmp_size), int(y - tmp_size)]
    br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]

    if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
        return

    size = 2 * tmp_size + 1
    x_range = np.arange(0, size, 1, np.float32)
    y_range = x_range[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x_range - x0)**2 + (y_range - y0)**2) / (2 * sigma**2))

    g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
    img_x = max(0, ul[0]), min(br[0], W)
    img_y = max(0, ul[1]), min(br[1], H)

    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

def draw_curve_region(points, heatmap, sigma=5, num_points=30):
    if len(points) < 2:
        return
    elif len(points) == 2:
        # 直线插值
        x1, y1 = points[0]
        x2, y2 = points[1]
        for alpha in np.linspace(0, 1, num_points):
            x = (1 - alpha) * x1 + alpha * x2
            y = (1 - alpha) * y1 + alpha * y2
            draw_gaussian(heatmap, (x, y), sigma)
        return

    x = [p[0] for p in points]
    y = [p[1] for p in points]
    k = min(3, len(points) - 1)

    try:
        tck, _ = splprep([x, y], s=0, k=k)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)

        for x_, y_ in zip(x_new, y_new):
            draw_gaussian(heatmap, (x_, y_), sigma=sigma)
    except Exception as e:
        print(f"⚠️ spline 拟合失败: {e}，fallback to point")
        for pt in points:
            draw_gaussian(heatmap, pt, sigma)

# ======= 主融合逻辑 =======
final_heatmap = np.zeros((H, W), dtype=np.float32)

for region_name, info in regions.items():
    indices = info["indices"]
    weight = info["weight"]
    region_points = [keypoints[i] for i in indices]

    region_heatmap = np.zeros_like(final_heatmap)

    if "eyebrow" in region_name:
        draw_curve_region(region_points, region_heatmap, sigma=5, num_points=spline_num_points)
    else:
        for pt in region_points:
            draw_gaussian(region_heatmap, pt, sigma=sigma)

    final_heatmap += region_heatmap * weight

# ======= 后处理 + 保存 =======
if normalize_output:
    final_heatmap = np.clip(final_heatmap, 0, 1)

colored = cv2.applyColorMap((final_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(image, 0.6, colored, 0.4, 0)

cv2.imwrite(output_path, overlay)
print(f"✅ 加权关键点热力图已保存到: {output_path}")