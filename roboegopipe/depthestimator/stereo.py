"""
# 伪代码示意
# 1. 计算双目外参
T_body_cam1 = 从你的数据解析
T_body_cam2 = 从你的数据解析
T_cam1_cam2 = inv(T_body_cam1) * T_body_cam2

# 2. 双目校正（使两图像行对齐）
R1, R2, P1, P2, Q = stereoRectify(
    K1, D1, K2, D2, 
    R_cam1_cam2, t_cam1_cam2, 
    image_size, alpha=0
)

# 3. 去畸变 + 校正
map1_x, map1_y = initUndistortRectifyMap(...)
map2_x, map2_y = initUndistortRectifyMap(...)

# 4. 立体匹配（SGBM 或 深度学习）
disparity = stereo_match(rectified_img1, rectified_img2)

# 5. 深度计算
depth = Q[2,3] / (disparity + Q[3,2])
"""

class StereoEstimator():
    def __init__():
