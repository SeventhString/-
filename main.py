import cv2
import numpy as np
import os

# 感知哈希算法

# 定义感知哈希
def phash(img):
    # step1：调整大小32x32
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)

    # step2:离散余弦变换
    img = cv2.dct(img)
    img = img[0:8, 0:8]
    sum = 0.
    hash_str = ''

    # step3:计算均值
    # avg = np.sum(img) / 64.0
    for i in range(8):
        for j in range(8):
            sum += img[i, j]
    avg = sum/64

    # step4:获得哈希值
    for i in range(8):
        for j in range(8):
            if img[i, j] > avg:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str

# 计算汉明距离
def hmdistance(hash1,hash2):
    num = 0
    assert len(hash1) == len(hash2)
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            num += 1
    return num

# 遍历图库
def getFileList(dir, Filelist, ext=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)
    return Filelist

org_img_folder = './org'
imglist = getFileList('pics_data', [], 'jpg')
names = dict()

def Compare():
    IMG = input("请输入图片路径:")
    img1 = cv2.imread(IMG)
    for jpg in imglist:
        img2 = cv2.imread(jpg)
        hash1 = phash(img1)
        hash2 = phash(img2)
        dist = hmdistance(hash1, hash2)
        names[dist] = jpg

Compare()
names_sorted = sorted(names.items(), key=lambda x: x[0])
for item in names_sorted:
    print(item)
    # 输出结果越小即越相似