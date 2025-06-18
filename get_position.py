import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("/home/hanna/桌面/vHeat/dataset/rafdb_split_singel/train/Anger/train_06522.jpg")

def onclick(event):
    print(f"你点击了位置: x={int(event.xdata)}, y={int(event.ydata)}")

plt.imshow(img)
plt.title("点击图像以获取像素位置")
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.show()