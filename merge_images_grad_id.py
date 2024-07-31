from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps


image_paths = [r'sjtu_grad_cert.png']
texts = [r'此毕业证复印件仅用于档案寄送']
text_color = (255, 0, 0)
font = ImageFont.truetype('simsun.ttc', 64)
# font = ImageFont.truetype('arial.ttf')

images = []

for image_path in image_paths:
    curr_image_path = Path(r'D:\Data\chicago\jiongjiong') / image_path
    # print(curr_image_path.as_posix())

    image = Image.open(curr_image_path.as_posix())
    # resized_height = 640
    # image = image.resize((int(resized_height * 1.618), resized_height))
    images.append(image)

    width, height = images[0].size

    new_image = Image.new(mode="RGB", size=(width, height))

    left = 0
    top = 0

    new_image.paste(image, box = (left, top))
    top += height


    draw = ImageDraw.Draw(new_image)

    text = texts[0]
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    left_top = ((width - text_width) // 2, (height - text_height) * 2 // 4)
    draw.text(left_top, text, text_color, font=font)

    # top, right, bottom, left
    # border = (2,) * 4
    # color = 'midnightblue'
    # new_image = ImageOps.expand(new_image, border=border, fill=color)

    new_image.show()
