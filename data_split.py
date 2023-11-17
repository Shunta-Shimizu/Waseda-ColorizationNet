import numpy as np
import glob
import cv2

# gray2rgb_datasets = glob.glob("/home/shimizu/waseda_color/dataset/rgb/*")

# train_len = int(len(gray2rgb_datasets)*0.9)
# test_len = len(gray2rgb_datasets)-train_len

# np.random.seed(0)
# np.random.shuffle(gray2rgb_datasets)
# train_rgbdata = gray2rgb_datasets[:train_len]
# test_rgbdata = gray2rgb_datasets[train_len:]

# for d in train_rgbdata:
#     name = d.split("/home/shimizu/waseda_color/dataset/rgb/")[1]
#     rgb_img = cv2.imread(d)
#     cv2.imwrite("/home/shimizu/waseda_color/dataset/train/rgb/"+name, rgb_img)
#     # gray_img = cv2.imread("/home/shimizu/waseda_color/dataset/train2/gray/"+name)
#     gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite("/home/shimizu/waseda_color/dataset/train/gray/"+name, gray_img)

# for d in test_rgbdata:
#     name = d.split("/home/shimizu/waseda_color/dataset/rgb/")[1]
#     rgb_img = cv2.imread(d)
#     cv2.imwrite("/home/shimizu/waseda_color/dataset/test/rgb/"+name, rgb_img)
#     # gray_img = cv2.imread("/home/shimizu/waseda_color/test"+name)
#     gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite("/home/shimizu/waseda_color/dataset/test/gray/"+name, gray_img)

