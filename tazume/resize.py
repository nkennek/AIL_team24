import cv2
import glob
import numpy as np

img_num = 0
for f in glob.glob("download_ladys/*.jpg"):
    img = cv2.imread(f)
    tmp = img[:, :]
    height, width = img.shape[:2]
    if(height > width):
	    size = height
	    limit = width
    else:
	    size = width
	    limit = height
    start = int((size - limit) / 2)
    fin = int((size + limit) / 2)
    new_img = cv2.resize(np.zeros((1, 1, 3), np.uint8), (size, size))
    if(size == height):
        new_img[:, start:fin] = tmp
    else:
	    new_img[start:fin, :] = tmp
    new_img = cv2.resize(new_img, (256,256))
    cv2.imwrite("train_ladys/image_{0:06d}.jpg".format(img_num), new_img)
    img_num += 1
    if img_num % 500 == 0:
        print(img_num)
    