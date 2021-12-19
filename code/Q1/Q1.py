import cv2
import numpy as np

def img1a2(path, n=3):
    '''
    :param path:图片地址
    :n: 图形形态学处理中算子的大小
    :return: 经过滤波、二值化后的**三通道**图像，可用来传进Edge()函数
    '''
    org = cv2.imread(path)
    img = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ret, binary = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)  # 二值化
    k = np.ones((n, n))  # 开运算算子,消除毛边
    open = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=5)
    # 显示二值化图像
    tmp = org.copy()
    tmp[:, :, 0], tmp[:, :, 1], tmp[:, :, 2] = open, open, open
    return tmp

def img3(path, th=3000):
    '''
    :param path: 图片地址
    :th: 所要删除的小区域的面积阈值，面积小于th个像素的小区域会被填充成黑色
    :return: 经过滤波、直方图均衡化、二值化、形态学操作后的**三通道**图像，可用来传进Edge()函数
    '''
    org = cv2.imread(path)
    img = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    equ = cv2.equalizeHist(img)
    ret, binary = cv2.threshold(equ, 254.5, 255, cv2.THRESH_BINARY)  # 二值化
    k = np.ones((2, 2))  # 闭运算算子
    close = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=5)
    contours, _ = cv2.findContours(close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))  # 计算轮廓包围的面积
    print(area)
    for k in range(len(area)):
        if area[k] <= th:
            cv2.fillPoly(close, [contours[k]], 0)

    tmp = org.copy()
    tmp[:, :, 0], tmp[:, :, 1], tmp[:, :, 2] = close, close, close
    return tmp

if __name__ == "__main__":
    # 图片1和2
    path1 = 'C:/Users/23687/Desktop/APMCM/2021 APMCM Problems/2021 APMCM Problem A/Annex 1/Pic1_1.bmp'
    path2 = 'C:/Users/23687/Desktop/APMCM/2021 APMCM Problems/2021 APMCM Problem A/Annex 1/Pic1_2.bmp'
    tmp1 = img1a2(path1)
    tmp2 = img1a2(path2)
    cv2.imshow('pic1_2', tmp2)
    cv2.imshow('pic1_1', tmp1)

    # 图片3
    path3 = 'C:/Users/23687/Desktop/APMCM/2021 APMCM Problems/2021 APMCM Problem A/Annex 1/Pic1_3.bmp'
    tmp3 = img3(path3)
    cv2.imshow('pic1_3', tmp3)

    k = np.ones((3, 3))  # 开运算算子,消除毛边

    org = cv2.imread(path3)
    img = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)
    filtered = cv2.bilateralFilter(img, 9, 75, 75)  # 滤波后
    equ = cv2.equalizeHist(filtered)  # 均衡化
    _, binary = cv2.threshold(equ, 254.5, 255, cv2.THRESH_BINARY)  # 二值化
    # binary = 255 - binary
    close = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=5)  # 闭操作
    contours, _ = cv2.findContours(close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))  # 计算轮廓包围的面积
    print(area)
    for k in range(len(area)):
        if area[k] <= 3000:
            C2 = cv2.fillPoly(close, [contours[k]], 0)
    cv2.imshow("gray", img)
    cv2.imshow("filtered", filtered)
    cv2.imshow("equ", equ)
    cv2.imshow("binary", 255-binary)
    cv2.imshow("close", 255-close)
    cv2.imshow("C2", 255-C2)

    cv2.imwrite("result.png", 255-C2)

    cv2.imwrite("Q1_3_filtered.png", filtered)
    cv2.imwrite("Q1_3_equ.png", equ)
    cv2.imwrite("Q1_3_binary.png", 255-binary)
    cv2.imwrite("Q1_3_close.png", 255-close)
    cv2.waitKey(0)