from pathlib import Path
from PIL import Image, ImageOps

dir_path = Path(r'D:\Data\images')
dir_path = Path(r'D:\Data\chicago\jiongjiong')

image_names = [
    # # r'图片_20240212210842.jpg',
    # # r'图片_20240212210906.jpg',
    # # r'图片_20240212210919.jpg',
    # # r'图片_20240212210927.jpg',

    # # r'IMG20240212220727.jpg',
    # # r'IMG20240212220800.jpg',
    # # r'IMG20240212220827.jpg',
    # # r'IMG20240212220910.jpg',

    # r'IMG20240212225754.jpg',
    # r'IMG20240212230132.jpg',
    # r'IMG20240212230220.jpg',
    # r'IMG20240212230302.jpg',
    # r'IMG20240212230348.jpg',
    # r'IMG20240212230424.jpg',
    # r'IMG20240212230457.jpg',
    # # r'IMG20240212230532.jpg',
    # r'IMG20240212230541.jpg',

    r'JiongjiongLi_transcript_ecnu_cn_1.jpg',
    r'JiongjiongLi_transcript_ecnu_cn_2.jpg',
]

image_list = []

for image_name in image_names:
    image_path = dir_path / image_name
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert('RGB')
    # image = image.resize((image.size[0] // 2,image.size[1] // 2),Image.ANTIALIAS)
    print(image.size)
    image_list.append(image)

# image_list[0].save(dir_path / r'I-129S.pdf',save_all=True, append_images=image_list[1:], optimize=True)

image_list[0].save(dir_path / r'JiongjiongLi_transcript_ecnu_cn.pdf',save_all=True, append_images=image_list[1:], optimize=True)
