"""生成 120 组合成图文样本用于 CLIP 检索实验"""
import os
import csv
from PIL import Image, ImageDraw

out_dir = r"C:\Users\text\Desktop\vit-from-scratch\data\clip_images"
os.makedirs(out_dir, exist_ok=True)

colors = {
    "红色": (255, 0, 0),
    "蓝色": (0, 0, 255),
    "绿色": (0, 200, 0),
    "橘色": (255, 120, 0),
    "黄色": (255, 255, 0),
    "紫色": (160, 50, 200),
    "粉色": (255, 150, 180),
    "棕色": (139, 69, 19),
    "灰色": (128, 128, 128),
    "青色": (0, 200, 200),
}
shapes = ["方块", "圆形", "三角形"]

rows = []
idx = 0
for shape in shapes:
    for color_name, rgb in colors.items():
        for i in range(4):  # 每种 4 个变体（为达 120 张）
            idx += 1
            img = Image.new('RGB', (224, 224), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)

            cx, cy = 112, 112
            size = 60 + i * 10  # 变体：不同大小
            r, g, b = rgb
            r2, g2, b2 = min(r+20, 255), min(g+20, 255), min(b+20, 255)

            if shape == "方块":
                draw.rectangle([cx-size, cy-size, cx+size, cy+size], fill=rgb, outline=(r2,g2,b2), width=5)
            elif shape == "圆形":
                draw.ellipse([cx-size, cy-size, cx+size, cy+size], fill=rgb, outline=(r2,g2,b2), width=5)
            elif shape == "三角形":
                draw.polygon([(cx, cy-size), (cx-size, cy+size), (cx+size, cy+size)], fill=rgb, outline=(r2,g2,b2))

            fname = f"{idx:04d}.jpg"
            img.save(os.path.join(out_dir, fname))
            caption = f"一个{color_name}的{shape}"
            rows.append([fname, caption])

# 写 metadata.csv
csv_path = os.path.join(out_dir, "metadata.csv")
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "caption"])
    writer.writerows(rows)

print(f"生成 {idx} 张图，metadata 保存到 {csv_path}")
print(f"前 5 条: {rows[:5]}")
