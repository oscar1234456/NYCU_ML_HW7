import numpy as np
import cv2
import os
from PIL import Image


def load_image(path):
    # the read order is important!
    pic_filenames = sorted(os.listdir(path))
    print(f"--Found {len(pic_filenames)} images--")
    labels = list()
    image_pixel = list()
    for pic_filename in pic_filenames:
        print(f"pic filename: {pic_filename}")
        label = int(pic_filename.split(".")[0][7:])
        labels.append(label)
        image = Image.open(path + "/" + pic_filename).resize((195, 231))
        image_pixel.append(list(image.getdata()))
    return np.array(image_pixel, dtype="float"), np.array(labels, dtype="uint8")


if __name__ == "__main__":
    image, label = load_image("../Yale_Face_Database/Training")
    print()
