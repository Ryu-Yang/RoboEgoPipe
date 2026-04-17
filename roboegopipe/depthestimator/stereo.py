"""
双目深度估计器

使用 OpenCV SGBM 进行立体匹配，计算深度图。

伪代码示意:
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

import cv2
import numpy as np
import logging
from scipy.spatial.transform import Rotation as R

log = logging.getLogger()


def creat_matrix_from_pose(pose):
    """
    从 [x, y, z, qx, qy, qz, qw] 创建 4x4 变换矩阵
    """
    pos = np.array(pose[:3])
    quat = np.array(pose[3:])  # [qx, qy, qz, qw]
    rot_mat = R.from_quat(quat).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = pos
    return T


def generate_ds_undistort_map(width, height, fu, fv, cu, cv, xi, alpha):
    """
    使用 DS (Double Sphere) 模型生成去畸变映射表。
    
    基于数值方法生成映射表，原理：对于每个理想像素，寻找一个畸变像素，
    使其投影后最接近该理想像素的归一化坐标。
    
    Args:
        width: 图像宽度
        height: 图像高度
        fu, fv: 焦距 (x, y 方向)
        cu, cv: 主点坐标
        xi: DS 模型参数 (unified parameter)
        alpha: DS 模型参数 (distortion parameter)
    
    Returns:
        map_x, map_y: 去畸变映射表
    """
    # 创建理想网格
    u_out, v_out = np.meshgrid(np.arange(width), np.arange(height))
    u_out = u_out.astype(np.float64)
    v_out = v_out.astype(np.float64)
    
    # 理想归一化坐标
    x = (u_out - cu) / fu
    y = (v_out - cv) / fv
    z = np.ones_like(x)
    
    # DS 模型计算
    d1 = np.sqrt(x * x + y * y + z * z)
    
    # 变体公式：交换 alpha 权重
    mz = (1.0 - alpha) * d1 + alpha * z
    
    d2 = np.sqrt(x * x + y * y + mz * mz)
    
    # 变体分母
    denominator = (1.0 - alpha) * d2 + alpha * (xi * d1 + z)
    denominator = np.clip(denominator, 1e-8, None)
    
    # 计算畸变图像坐标
    u_in = fu * x / denominator + cu
    v_in = fv * y / denominator + cv
    
    return u_in.astype(np.float32), v_in.astype(np.float32)


class StereoEstimator:
    def __init__(
        self,
        num_disparities=64,
        block_size=11,
        min_disparity=0,
        uniqueness_ratio=10,
        speckle_window_size=100,
        speckle_max_size=1000,
        pre_filter_cap=31,
        p1=8 * 11 * 11,
        p2=32 * 11 * 11,
    ):
        """
        初始化 SGBM 立体匹配器参数
        """
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.min_disparity = min_disparity
        self.uniqueness_ratio = uniqueness_ratio
        self.speckle_window_size = speckle_window_size
        self.speckle_max_size = speckle_max_size
        self.pre_filter_cap = pre_filter_cap
        self.p1 = p1
        self.p2 = p2

        # 标定状态
        self._calibrated = False
        self._rect_maps = {}  # 存储左右相机的校正映射
        self._Q = None  # 重投影矩阵
        self._stereo_matcher = None
        self._is_ds_model = False  # 是否使用 DS 畸变模型

        self._setup_stereo_matcher()

    def _setup_stereo_matcher(self):
        """初始化 SGBM 立体匹配器"""
        self._stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=self.p1,
            P2=self.p2,
            disp12MaxDiff=1,
            preFilterCap=self.pre_filter_cap,
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window_size,
            speckleRange=self.speckle_max_size,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def calibrate(self, K1, D1, T_b_c1, K2, D2, T_b_c2, image_size, distortion_model=""):
        """
        标定双目系统，计算校正变换

        Args:
            K1: 左相机内参矩阵 (3x3) 或 [fx, fy, cx, cy, xi, alpha] (DS模型)
            D1: 左相机畸变系数
            T_b_c1: 左相机相对于 body 的位姿 [x, y, z, qx, qy, qz, qw]
            K2: 右相机内参矩阵 (3x3) 或 [fx, fy, cx, cy, xi, alpha] (DS模型)
            D2: 右相机畸变系数
            T_b_c2: 右相机相对于 body 的位姿 [x, y, z, qx, qy, qz, qw]
            image_size: 图像尺寸 (width, height)
            distortion_model: 畸变模型类型 ("ds" 表示 Double Sphere)
        """
        log.info("🔧 开始双目系统标定...")

        # 转换为 numpy 数组
        K1 = np.array(K1, dtype=np.float64)
        K2 = np.array(K2, dtype=np.float64)
        D1 = np.array(D1, dtype=np.float64)
        D2 = np.array(D2, dtype=np.float64)

        # 将扁平的内参矩阵 reshape 为 3x3
        if K1.shape == (9,):
            K1 = K1.reshape(3, 3)
        if K2.shape == (9,):
            K2 = K2.reshape(3, 3)

        # 确保畸变系数是正确形状
        if D1.ndim > 1:
            D1 = D1.flatten()
        if D2.ndim > 1:
            D2 = D2.flatten()

        # 检测是否使用 DS 模型
        # DS 模型: D 包含 [fu, fv, cu, cv, xi, alpha] (6个元素)
        # 或者 distortion_model 包含 "ds" / "double_sphere"
        is_ds_model = (
            distortion_model.lower() in ("ds", "double_sphere") or
            len(D1) == 6 and len(D2) == 6
        )
        
        self._is_ds_model = is_ds_model
        
        if is_ds_model:
            log.info("📷 检测到 DS (Double Sphere) 畸变模型")
            # 从 D 数组提取 DS 参数
            fu1, fv1, cx1, cy1, xi1, alpha1 = D1
            fu2, fv2, cx2, cy2, xi2, alpha2 = D2
            
            # 使用 DS 参数构建内参矩阵
            K1 = np.array([
                [fu1, 0, cx1],
                [0, fv1, cy1],
                [0, 0, 1]
            ], dtype=np.float64)
            K2 = np.array([
                [fu2, 0, cx2],
                [0, fv2, cy2],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # 保存 DS 参数用于后续去畸变
            self._ds_params = {
                "left": {"fu": fu1, "fv": fv1, "cu": cx1, "cv": cy1, "xi": xi1, "alpha": alpha1},
                "right": {"fu": fu2, "fv": fv2, "cu": cx2, "cv": cy2, "xi": xi2, "alpha": alpha2},
            }
            
            # 对于 stereoRectify，使用零畸变
            D1 = np.zeros(5, dtype=np.float64)
            D2 = np.zeros(5, dtype=np.float64)
        else:
            self._ds_params = None
            # OpenCV 要求畸变系数长度为 4, 5, 8, 12 或 14
            # 如果长度不合法，回退到无畸变
            valid_d_lengths = {4, 5, 8, 12, 14}
            if len(D1) not in valid_d_lengths:
                log.warning(f"⚠️ 左相机畸变系数长度 {len(D1)} 无效，回退到无畸变")
                D1 = np.zeros(5, dtype=np.float64)
            if len(D2) not in valid_d_lengths:
                log.warning(f"⚠️ 右相机畸变系数长度 {len(D2)} 无效，回退到无畸变")
                D2 = np.zeros(5, dtype=np.float64)

        # 计算 T_cam1_cam2 = inv(T_body_cam1) @ T_body_cam2
        T_body_cam1 = creat_matrix_from_pose(T_b_c1)
        T_body_cam2 = creat_matrix_from_pose(T_b_c2)
        T_cam1_cam2 = np.linalg.inv(T_body_cam1) @ T_body_cam2

        # 提取旋转和平移
        R_cam1_cam2 = T_cam1_cam2[:3, :3]
        t_cam1_cam2 = T_cam1_cam2[:3, 3]

        log.debug(f"T_cam1_cam2:\n{T_cam1_cam2}")
        log.debug(f"旋转矩阵 R:\n{R_cam1_cam2}")
        log.debug(f"平移向量 t:\n{t_cam1_cam2}")

        # 双目校正
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            K1, D1, K2, D2, image_size, R_cam1_cam2, t_cam1_cam2, alpha=0
        )

        # 计算去畸变校正映射表
        width, height = image_size
        
        if self._is_ds_model:
            # DS 模型：先生成去畸变映射，然后与立体校正映射组合
            ds_map_left = generate_ds_undistort_map(width, height, **self._ds_params["left"])
            ds_map_right = generate_ds_undistort_map(width, height, **self._ds_params["right"])
            
            # 保存 DS 去畸变映射和立体校正映射
            self._rect_maps = {
                "left": {
                    "ds_map_x": ds_map_left[0],
                    "ds_map_y": ds_map_left[1],
                    "rect_map_x": None,  # 将在 rectify_image 中动态计算
                    "rect_map_y": None,
                },
                "right": {
                    "ds_map_x": ds_map_right[0],
                    "ds_map_y": ds_map_right[1],
                    "rect_map_x": None,
                    "rect_map_y": None,
                },
            }
            # 保存立体校正参数
            self._stereo_rect_params = {
                "K1": K1, "D1": np.zeros(5), "R1": R1, "P1": P1,
                "K2": K2, "D2": np.zeros(5), "R2": R2, "P2": P2,
                "image_size": image_size,
            }
        else:
            # 标准 OpenCV 模型
            map1_x, map1_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
            map2_x, map2_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
            
            self._rect_maps = {
                "left": {"map_x": map1_x, "map_y": map1_y},
                "right": {"map_x": map2_x, "map_y": map2_y},
            }

        self._Q = Q
        self._calibrated = True
        self._image_size = image_size
        self._P1 = P1
        self._P2 = P2
        self._R1 = R1
        self._R2 = R2
        self._valid_roi = validPixROI1
        self._K1 = K1
        self._K2 = K2

        log.info("✅ 双目系统标定完成")
        return True

    def get_rectified_params(self):
        """
        获取校正后的相机参数

        Returns:
            dict: 包含校正后的 K, T_b_c_correction_R, width, height
        """
        if not self._calibrated:
            raise RuntimeError("请先调用 calibrate() 方法进行标定")

        # 从 P1 提取校正后的内参
        K_rect = self._P1[:3, :3]

        # 校正旋转（用于更新 T_b_c）
        R1 = self._R1

        # 有效图像区域
        roi = self._valid_roi
        if roi is not None:
            x, y, w, h = roi
            width = w
            height = h
        else:
            width = self._image_size[0]
            height = self._image_size[1]

        return {
            "K_rect": K_rect,
            "R_rect": R1,
            "width": int(width),
            "height": int(height),
        }

    def _rectify_image(self, image, side):
        """
        对单张图像应用去畸变和行对齐校正

        Args:
            image: 输入图像 (H, W, C) 或 (H, W)
            side: "left" 或 "right"
        """
        if not self._calibrated:
            raise RuntimeError("请先调用 calibrate() 方法进行标定")

        if side not in self._rect_maps:
            raise ValueError(f"无效的 side: {side}，应为 'left' 或 'right'")

        maps = self._rect_maps[side]

        if self._is_ds_model:
            # DS 模型：先应用去畸变，再应用立体校正
            # 第一步：去畸变
            if image.ndim == 3:
                channels = []
                for c in range(image.shape[2]):
                    channel = cv2.remap(
                        image[:, :, c], 
                        maps["ds_map_x"], maps["ds_map_y"], 
                        cv2.INTER_LINEAR, cv2.BORDER_CONSTANT
                    )
                    channels.append(channel)
                undistorted = np.stack(channels, axis=2)
            else:
                undistorted = cv2.remap(
                    image, maps["ds_map_x"], maps["ds_map_y"], 
                    cv2.INTER_LINEAR, cv2.BORDER_CONSTANT
                )
            
            # 第二步：立体校正（需要计算校正映射）
            rect_params = self._stereo_rect_params
            if side == "left":
                R_rect = rect_params["R1"]
                P_rect = rect_params["P1"]
                K_rect = rect_params["K1"]
                D_rect = rect_params["D1"]
            else:
                R_rect = rect_params["R2"]
                P_rect = rect_params["P2"]
                K_rect = rect_params["K2"]
                D_rect = rect_params["D2"]
            
            # 计算立体校正映射
            rect_map_x, rect_map_y = cv2.initUndistortRectifyMap(
                K_rect, D_rect, R_rect, P_rect, 
                self._image_size, cv2.CV_32FC1
            )
            
            # 应用立体校正
            if undistorted.ndim == 3:
                channels = []
                for c in range(undistorted.shape[2]):
                    channel = cv2.remap(
                        undistorted[:, :, c], 
                        rect_map_x, rect_map_y, 
                        cv2.INTER_LINEAR
                    )
                    channels.append(channel)
                return np.stack(channels, axis=2)
            else:
                return cv2.remap(undistorted, rect_map_x, rect_map_y, cv2.INTER_LINEAR)
        else:
            # 标准 OpenCV 模型
            # 处理灰度图
            if image.ndim == 2:
                rectified = cv2.remap(image, maps["map_x"], maps["map_y"], cv2.INTER_LINEAR)
                return rectified

            # 处理彩色图
            channels = []
            for c in range(image.shape[2]):
                channel = cv2.remap(image[:, :, c], maps["map_x"], maps["map_y"], cv2.INTER_LINEAR)
                channels.append(channel)
            return np.stack(channels, axis=2)

    def compute_disparity(self, img_left, img_right):
        """
        计算视差图

        Args:
            img_left: 左校正图像 (灰度或彩色)
            img_right: 右校正图像 (灰度或彩色)

        Returns:
            disparity: 视差图 (16位有符号整数，需要除以16得到真实视差)
        """
        if not self._calibrated:
            raise RuntimeError("请先调用 calibrate() 方法进行标定")

        # 转换为灰度图
        if img_left.ndim == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left

        if img_right.ndim == 3:
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_right = img_right

        # 确保数据类型为 uint8
        if gray_left.dtype != np.uint8:
            gray_left = (gray_left * 255).astype(np.uint8)
        if gray_right.dtype != np.uint8:
            gray_right = (gray_right * 255).astype(np.uint8)

        # 计算视差
        disparity = self._stereo_matcher.compute(gray_left, gray_right)

        return disparity

    def compute_depth(self, img_left, img_right):
        """
        计算单帧深度图

        Args:
            img_left: 左校正图像
            img_right: 右校正图像

        Returns:
            depth: 深度图 (float32)，无效深度为 0 或 inf
        """
        disparity = self.compute_disparity(img_left, img_right)

        # 将 16 位视差转换为浮点视差
        disparity_float = disparity.astype(np.float32) / 16.0

        # 使用 Q 矩阵计算深度
        # depth = Q[2,3] / (disparity + Q[3,2])
        # 注意：OpenCV 的 reprojectImageTo3D 也可以，但直接计算更高效
        depth = np.zeros_like(disparity_float)

        # 只处理有效视差（> 0）
        valid_mask = disparity_float > 0
        if np.any(valid_mask):
            depth[valid_mask] = self._Q[2, 3] / (disparity_float[valid_mask] + self._Q[3, 2])

        # 处理异常值
        max_valid_depth = 10.0  # 10 米最大深度
        depth[depth > max_valid_depth] = 0.0
        depth[depth < 0] = 0.0

        return depth

    def compute_depth_batch(self, images_left, images_right, timestamps_left, timestamps_right):
        """
        批量计算深度，返回与 images 相同结构的数据

        Args:
            images_left: 左相机图像列表
            images_right: 右相机图像列表
            timestamps_left: 左相机时间戳列表
            timestamps_right: 右相机时间戳列表

        Returns:
            depth_data: {"depth_maps": [...], "timestamps": [...]}
        """
        if not self._calibrated:
            raise RuntimeError("请先调用 calibrate() 方法进行标定")

        log.info(f"🔄 开始批量计算深度，左相机 {len(images_left)} 帧，右相机 {len(images_right)} 帧...")

        depth_maps = []
        depth_timestamps = []

        # 对左右相机图像按时间戳对齐
        # 使用最近邻匹配
        ts_right_array = np.array(timestamps_right)

        for i, (img_left, ts_left) in enumerate(zip(images_left, timestamps_left)):
            # 找到最近的右相机帧
            ts_diff = np.abs(ts_right_array - ts_left)
            best_right_idx = np.argmin(ts_diff)
            min_ts_diff = ts_diff[best_right_idx]

            # 容忍 100ms 的时间差
            if min_ts_diff > 100_000_000:
                log.debug(f"⚠️ 帧 {i} 时间差过大 ({min_ts_diff / 1e6:.0f}ms)，跳过")
                continue

            img_right = images_right[best_right_idx]
            ts_used = ts_left

            try:
                # 校正图像
                rect_left = self._rectify_image(img_left, "left")
                rect_right = self._rectify_image(img_right, "right")

                # 计算深度
                depth = self.compute_depth(rect_left, rect_right)
                depth_maps.append(depth)
                depth_timestamps.append(ts_used)

                if (i + 1) % 10 == 0:
                    log.info(f"  进度: {i + 1}/{len(images_left)} 帧")

            except Exception as e:
                log.warning(f"⚠️ 计算帧 {i} 深度失败: {e}")
                continue

        log.info(f"✅ 批量深度计算完成，共 {len(depth_maps)} 帧")

        return {
            "depth_maps": depth_maps,
            "timestamps": depth_timestamps,
        }

    def calibrate_from_camera_info(self, cam_info1, cam_info2, image_size):
        """
        从相机信息字典中直接标定

        Args:
            cam_info1: 左相机信息字典，包含 K, D, T_b_c
            cam_info2: 右相机信息字典，包含 K, D, T_b_c
            image_size: 图像尺寸 (width, height)
        """
        K1 = cam_info1.get("K", [])
        D1 = cam_info1.get("D", [])
        T_b_c1 = cam_info1.get("T_b_c", [])

        K2 = cam_info2.get("K", [])
        D2 = cam_info2.get("D", [])
        T_b_c2 = cam_info2.get("T_b_c", [])
        
        # 获取畸变模型类型
        distortion_model = cam_info1.get("distortion_model", "")

        return self.calibrate(K1, D1, T_b_c1, K2, D2, T_b_c2, image_size, distortion_model)