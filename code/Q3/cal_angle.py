import numpy as np
def clockwise_angle(v1, v2):
    x1,y1 = v1
    x2,y2 = v2
    dot = x1*x2+y1*y2
    det = x1*y2-y1*x2
    theta = np.arctan2(det, dot)
    theta = theta if theta>0 else 2*np.pi+theta
    return theta

def GetClockAngle(v1, v2):
    # 2个向量模的乘积
    TheNorm = np.linalg.norm(v1)*np.linalg.norm(v2)
    # 叉乘
    rho = np.rad2deg(np.arcsin(np.cross(v1, v2)/TheNorm))
    # 点乘
    theta = np.rad2deg(np.arccos(np.dot(v1,v2)/TheNorm))
    if rho < 0:
        return -theta
    else:
        return theta

if __name__ =="__main__":
    start_point = np.array([1184, 666])
    center_point = np.array([1180, 693])
    end_point = np.array([1180, 719])
    v1 = start_point - center_point
    v2 = end_point - center_point

    theta = GetClockAngle(v1, v2)
    print(theta)
    print(v1)
    print(v2)
    """
    从x轴正方向到y轴正方形的方向是旋转/扫描的正方向
    给出曲线段的起点、终点和中心点，计算出两个向量，从而计算出向量旋转角度
    """
    # a = [0, 1]
    # b = [1, 0]
    # c = [-1, 0]
    # d = [0, -1]
    # e = [-1, -1]
    # f = [1, -1]
    # g = [1, 1]
    # h = [-1, 1]
