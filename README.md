# 2021APMCM-A（2等奖）
**背景**：传统卡尺测量技术难以及时、高效地完成尺寸检测的任务。机器视觉的特点是**自动化、客观、非接触和高精度**，可以解决上述问题。因此，**工业环境下数字图像边缘的分析和应用**有重要的研究价值。

针对问题1，建立了一种基于Partial Area Effect的图像亚像素边缘检测方法。首先，使用双边滤波、二值分割、形态学操作对图像进行预处理。其中，对Pic1_3进行直方图均衡化，并根据轮廓包围的像素数识别反光区域以消除反光的影响。随后进行亚像素边缘检测。最后，使用最近邻搜索对每个轮廓的点进行排序，得到有序的边缘轮廓数据。

针对问题2，建立了相机的标定和校准模型，并建立从像素尺寸到实际物理尺寸的转换关系。首先，找出三张校准板照片中圆心的对应关系，使用张正友法对相机进行标定。计算得到相机的内参矩阵、外参矩阵和畸变系数。其次，对校准板图像进行正畸，将校准板中的圆作为标准件进行系统标定，计算得到像素当量，求出Pic2_4中各轮廓的实际尺寸，total contours length=162.746596mm。

针对问题3，建立了一种基于滑动窗口拟合圆半径的轮廓曲线自动分割方法，和基于最小二乘法的曲线段拟合方法。首先，按照轮廓点的排列顺序，选取连续的20个点作为滑动窗口，进行圆的拟合，根据拟合圆的曲率半径将轮廓分割为直线段和曲线段。随后，采用基于直接最小二乘的椭圆拟合法对曲线段进行拟合，将长短轴差距在5%以内的轮廓段标记为圆弧段。最后，根据起点、终点和中心点计算扫描角度，得到两组Coutour Segmentation Data。


**Q1 边缘提取效果**：

![image](https://user-images.githubusercontent.com/75946871/146675229-5514ee53-5c76-4673-b9ef-9c458a1a1c2c.png)

![image](https://user-images.githubusercontent.com/75946871/146675246-64c3c37f-bf4f-4427-8839-9370ef6c8dc7.png)

**Q3轮廓分割效果**：

![image](https://user-images.githubusercontent.com/75946871/146675311-4cd3acbb-ca30-486a-9124-d588b473b680.png)

![image](https://user-images.githubusercontent.com/75946871/146675313-fdd20324-294f-4cca-8cf3-a10f8b9cbf8d.png)

*利用直接最小二乘法进行椭圆拟合*：

![image](https://user-images.githubusercontent.com/75946871/146675324-739bea35-ad68-4596-b69a-adbbbbd2e973.png)

**轮廓分段效果图**：

![image](https://user-images.githubusercontent.com/75946871/146675333-93f21d9f-a8b2-4f1a-aeeb-123d3faefb66.png)

![image](https://user-images.githubusercontent.com/75946871/146675335-f4c8bb8c-c0cc-4f09-9a48-c0dc42b1732a.png)

**分段点的分布**：

![image](https://user-images.githubusercontent.com/75946871/146675345-54831ccc-5e07-420e-9af2-34501fb525f7.png)

![image](https://user-images.githubusercontent.com/75946871/146675352-3ea33e4a-6392-48a3-b8d4-09a193bf840b.png)
