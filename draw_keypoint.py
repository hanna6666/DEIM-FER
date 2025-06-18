import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import os

# ======= 路径设置（保持不变）=======
keypoint_file = "/home/hanna/桌面/rafdb_basic./Annotation/auto/train_09209_auto_attri.txt"
image_path = "/home/hanna/桌面/rafdb_basic./Image/original/train_09209.jpg"
output_path = "/home/hanna/桌面/DEIM/train_09209_kpt_mask_bezier.jpg"

# ======= 加载图像 =======
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"❌ 图像文件未找到: {image_path}")
mask = np.zeros_like(image)

# ======= 读取关键点 =======
keypoints = []
with open(keypoint_file, 'r') as f:
    for line in f:
        x, y = map(float, line.strip().split())
        keypoints.append((int(x), int(y)))

if len(keypoints) != 37:
    raise ValueError(f"❌ 关键点数量应为 37，当前为 {len(keypoints)}")

# ======= 定义区域索引 =======
regions = {
    "left_eyebrow": list(range(0, 3)),      # 0,1,2
    "right_eyebrow": list(range(3, 6)),     # 3,4,5
    "nose": list(range(6, 8)),             # 6~12
    "nose_blow": list(range(8, 13)),             # 6~12
    "left_eye": list(range(13, 19)),        # 13~18
    "right_eye": list(range(19, 25)),       # 19~24
    "mouth": list(range(25, 37))            # 25~36
}

# ======= 曲线绘制函数（支持自动降阶） =======
def draw_curve(points, mask, closed=False, color=(0, 255, 0), thickness=1):
    if len(points) < 2:
        return
    elif len(points) <= 3:
        for i in range(len(points) - 1):
            cv2.line(mask, points[i], points[i + 1], color, thickness)
        if closed and len(points) > 2:
            cv2.line(mask, points[-1], points[0], color, thickness)
        return

    points_np = np.array(points, dtype=np.float32).T
    k = min(3, len(points) - 1)
    try:
        tck, u = splprep(points_np, s=0, per=closed, k=k)
        u_fine = np.linspace(0, 1, 100)
        x_fine, y_fine = splev(u_fine, tck)

        for i in range(len(x_fine) - 1):
            pt1 = (int(x_fine[i]), int(y_fine[i]))
            pt2 = (int(x_fine[i + 1]), int(y_fine[i + 1]))
            cv2.line(mask, pt1, pt2, color, thickness)
    except Exception as e:
        print(f"⚠️ 区域绘制失败（可能是点太少），跳过：{e}")

# ======= 绘制所有区域 =======
for name, indices in regions.items():
    pts = [keypoints[i] for i in indices]
    closed = name in ["left_eye", "right_eye", "mouth"]
    draw_curve(pts, mask, closed=closed)

# ======= 合成并保存，不弹窗 =======
overlay = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
cv2.imwrite(output_path, overlay)
print(f"✅ 保存成功: {output_path}")