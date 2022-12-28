import os
import glob
from PIL import Image

files = glob.glob(
    r"C:/Users/freet/Desktop/3학년2학기/인공지능/과제/꽃 분류/train_data/*/*.jpg")

for f in files:
    img = Image.open(f)
    img_resize = img.resize((224, 224))
    title, ext = os.path.splitext(f)
    img_resize.save(title + ext)
