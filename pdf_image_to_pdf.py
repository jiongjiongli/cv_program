from pathlib import Path
from PIL import Image, ImageOps

dir_path = Path(r'D:\Data\images')


import pdfplumber

with pdfplumber.open(r"D:\Data\Jingjing Qin.pdf") as pdf:
    first_page = pdf.pages[0]
    print(first_page.images[0])

import fitz

input_pdf = r"D:\Data\Jingjing Qin.pdf"
output_pdf = r"D:\Data\Jingjing Qin letter.pdf"

doc = fitz.open(input_pdf)

for page_index in range(len(doc)): # iterate over pdf pages
    page = doc[page_index] # get the page
    image_list = page.get_images()
    page.delete_image(image_list[-1][0])

doc.save(output_pdf)



for page in doc:
    img_list = page.get_images()
    for img in img_list:
        page.delete_image(img[0])

doc.save(output_pdf)
