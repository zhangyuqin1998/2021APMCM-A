# -*- coding:utf-8 -*-
import cv2
import numpy as np
import time
import os
import pickle
import math


def calibrateCamera(file_path, w, h):
    """
    file_path: 存标定图片的文件夹，但是这个读取方式有一个缺点，文件夹内只能有图片，不能有别的类型的文件，否则会报错
    w: 棋盘格角点每行数量
    h: 棋盘格角点每列数量
    RETURN:重投影参数,内参矩阵,畸变系数,旋转向量,平移向量
    """
    distance = 2  # 相邻圆心的间距
    obj_points = np.zeros(((w * h), 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)  # 将纯0数组进行编码，编码代表每一个角点的位置信息，例如[0., 0., 0.],[1., 0., 0.]
    obj_points = np.reshape(obj_points, (w * h, 1, 3))  # 将位置信息矩阵变为w*h个1行三列的矩阵
    # 计算棋盘格内角点的三维坐标及其在图像中的二维坐标
    all_obj_points = []  # 这两个空数组很关键，如果是一张图片进行标定代码测试，这个也需要创建，如果没有，会一直报错
    all_points = []

    for file_name in os.listdir(file_path):

        img = cv2.imread(file_path + '/' + file_name)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图
        begin = time.time()
        ret, corners1 = cv2.findCirclesGrid(img_gray, (w, h))  # 寻找内角点
        if ret:  # 如果寻找到足够数量的内焦点
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # cv2.cornerSubPix(img_gray, corners1, (5, 5), (-1, -1), criteria)  # 细化内角点
            end = time.time()
            print((end - begin) / 60)
            img_h, img_w = img.shape[:2]  # 获取图像尺寸
            all_obj_points.append(distance * obj_points)  # 计算三维坐标
            all_points.append(corners1)  # 计算二维坐标
        else:
            end = time.time()
            print((end - begin) / 60)
            img_h, img_w = img.shape[:2]
        cv2.drawChessboardCorners(img, (20, 15), corners1, ret)  # 记住，OpenCV的绘制函数一般无返回值
        '''展示标定图片'''
        cv2.imshow(f'img{file_name}', img)
        cv2.imwrite(f'biaoding{file_name}.png', img)

    ret, camara_matrix, distcoeffs, rvecs, tvecs = cv2.calibrateCamera(all_obj_points, all_points,
                                                                       (img_w, img_h), None, None)
    # cv2.waitKey(0)
    return ret, camara_matrix, distcoeffs, rvecs, tvecs


def imgQ2(path):
    org = cv2.imread(img_path)
    img = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    ret, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)  # 二值化
    k = np.ones((3, 3))  # 闭运算算子
    close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=5)
    tmp = org.copy()
    tmp[:, :, 0], tmp[:, :, 1], tmp[:, :, 2] = close, close, close
    return tmp, close


if __name__ == "__main__":
    # 存图片的文件夹，但是这个读取方式有一个缺点，文件夹内只能有图片，不能有别的类型的文件，否则会报错
    file_path = 'C:/Users/23687/Desktop/APMCM/2021 APMCM Problems/2021 APMCM Problem A/Annex 2/pic/'
    img_path = 'C:/Users/23687/Desktop/APMCM/2021 APMCM Problems/2021 APMCM Problem A/Annex 2/Pic2_4.bmp'

    #计算相机参数，然后存起来
    # ret, camara_matrix, distcoeffs, rvecs, tvecs = calibrateCamera(file_path, 20, 15)

    print("内参矩阵: \n{}".format(camara_matrix))
    print("畸变系数: \n{}".format(distcoeffs))
    print("旋转向量：\n{}".format(rvecs))
    print("平移向量：\n{}".format(tvecs))
    print("重投影误差: \n{}".format(ret))  #0.8454487862566559
    cv2.waitKey(0)

    save_dict = {'内参矩阵': camara_matrix, '畸变系数': distcoeffs, '旋转向量': rvecs, '平移向量': tvecs}
    with open('calibrate_camera.p', 'wb') as f:
        pickle.dump(save_dict, f)
    #
    #读取相机参数。如果未进行过相机参数的标定，需要先计算相机参数并记录，才能进行读取
    with open('calibrate_camera.p', 'rb') as f:
        load_dict = pickle.load(f)
    camara_matrix, distcoeffs, rvecs, tvecs = load_dict['内参矩阵'], load_dict['畸变系数'], load_dict['旋转向量'], load_dict['平移向量']
    print("内参矩阵: \n{}".format(camara_matrix))
    print("畸变系数: \n{}".format(distcoeffs))
    print("旋转向量：\n{}".format(rvecs))
    print("平移向量：\n{}".format(tvecs))
    print("重投影误差: \n{}".format(ret))  #0.8454487862566559

    tmp, binary = imgQ2(img_path)
    undistorted = cv2.undistort(tmp, camara_matrix, distcoeffs, None, camara_matrix)
    cv2.imshow("正畸前", tmp)
    cv2.imshow("正畸后", undistorted)
    cv2.imwrite("tmp.png", tmp)
    cv2.imwrite("ed.png", undistorted)

    cv2.waitKey(0)
    binary = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    binary = np.array(binary)
    print(binary.shape)
    # img = 255 - img
    contours, hierarchy = cv2.findContours(255 - binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    Len = []
    for i in range(len(contours)):
        Len.append(cv2.arcLength(contours[i], True))  # 计算轮廓的周长
    print(f"周长是{Len}")
    color = [[255, 0, 0],
             [0, 255, 0],
             [0, 0, 255],
             [255, 255, 0],
             [0, 255, 255], ]

    tmp4 = np.zeros(tmp.shape)
    tmp4 = 255 - tmp4
    tmp = cv2.drawContours(tmp4, [contours[6]], -1, color[3], cv2.FILLED)
    cv2.imshow('draw', tmp4)
    cv2.imwrite('draw.png', tmp4)
    # cv2.imwrite('pic1_3.png', tmp3)
    cv2.imwrite('drawimg.png', binary)
    cv2.imshow('binary', binary)
    cv2.waitKey(0)


    """==========================DEMO=================================="""

    iii = cv2.imread('C:\\Users\\23687\\Desktop\\APMCM\\2021 APMCM Problems\\2021 APMCM Problem A\\Annex 2\\pic\\Pic2_1.bmp')
    gray = cv2.cvtColor(iii, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)  # 二值化
    binary = 255 - binary
    dst = cv2.undistort(binary, camara_matrix, distcoeffs, None, camara_matrix)
    contours, _ = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) <= 1000:
            area.append(cv2.contourArea(contours[i]))  # 计算轮廓包围的面积
    clr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    ret, tmp_ = cv2.threshold(clr, 255, 255, cv2.THRESH_BINARY)  # 二值化
    contours = contours[:-2]
    r = cv2.drawContours(tmp_, contours, -1, (0, 0, 255), 2)
    cv2.imshow("r",r)
    cv2.imwrite('circles_tmp.png', r)

    diameter = [math.sqrt(4*a/math.pi) for a in area] #计算直径
    print(diameter)
    print(f"直径均值是{np.mean(diameter)}像素")  #28.340235894027366 28.350347660448907 28.398006459934987
    print(f"直径方差是{np.var(diameter, ddof = 1)}像素")
    cv2.waitKey(0)

    """****
    由此，我们知道了像素大小和实际坐标系长度的对应关系，即：28.36286334像素对应实际坐标系中的1mm
    1像素对应实际坐标系中的0.035257371mm
    """
