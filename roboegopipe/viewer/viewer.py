import time
import logging
import rerun as rr
import numpy as np
import cv2

from roboegopipe.viewer.camera import create_camera_frustum, compute_camera_world_pose

log = logging.getLogger()

# MediaPipe Hand Landmarks Connections
# 21 landmarks per hand, indexed 0-20
HAND_CONNECTIONS = [
    # Wrist to thumb base
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Wrist to index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Wrist to middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Wrist to ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Wrist to pinky finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17),
]

# Hand landmark names for reference
HAND_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

class Viewer():
    def __init__(self):
        rr.init("RoboEgoPipe Viewer", spawn=True)
            
        # 为每个相机创建可视化
        self.camera_colors = {
            "link_camera0": [255, 100, 100],  # 红色
            "link_camera1": [100, 255, 150],  # 绿色
            "link_camera2": [100, 200, 255],  # 蓝色
            "link_camera3": [100, 200, 255],  # 蓝色
            "link_camera4": [100, 255, 150],  # 绿色
            "link_camera5": [255, 100, 100],  # 红色
        }
        
        self.default_colors = [255, 200, 100],  # 橙色
        
        self.traj_colors = {
            "eef_pose": [255, 100, 100],  # 红色
            "relative_eef_pose": [100, 200, 255],  # 蓝色
        }# 为每个轨迹创建不同的颜色

        # 创建3D视图
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # 添加坐标系轴
        rr.log("world/axes", rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        ), static=True)

        log.info("🚀 Rerun可视化已启动。\n")

    def _set_timestamp(self, timestamp_ns):
        """
        设置Rerun的时间戳
        
        Args:
            timestamp_ns: 纳秒时间戳
        """
        timestamp_seconds = timestamp_ns / 1e9
        
        rr.set_time("timestamp", timestamp=timestamp_seconds)

    def flush(self):
        rr.flush() 

    def _preprocess_image(self, image: np.ndarray, height: int, width: int) -> np.ndarray | None:
        """
        将图像转换为 Rerun 兼容的格式 (HWC, uint8)
        
        Args:
            image: 输入图像数据，可以是 1D (扁平), 2D (灰度), 或 3D (彩色/CHW/HWC)
            height: 期望的图像高度
            width: 期望的图像宽度
        """
        try:
            # 1. 确保是 numpy 数组
            if not isinstance(image, np.ndarray):
                image = np.array(image)

            # 2. 处理 1D 数组 (扁平化数据)
            if image.ndim == 1:
                total_pixels = height * width
                if len(image) == total_pixels:
                    # 假设是单通道灰度图
                    image = image.reshape((height, width, 1))
                elif len(image) == total_pixels * 3:
                    # 假设是三通道 RGB/BGR
                    image = image.reshape((height, width, 3))
                elif len(image) == total_pixels * 4:
                    # 假设是四通道 RGBA/BGRA
                    image = image.reshape((height, width, 4))
                else:
                    log.warning(f"1D image length {len(image)} does not match expected dimensions {height}x{width} for 1, 3, or 4 channels.")
                    return None

            # 3. 处理数据类型：转换为 uint8
            # 注意：reshape 后 dtype 不变，所以在这里统一处理 dtype
            if image.dtype != np.uint8:
                if image.dtype == np.float32 or image.dtype == np.float64:
                    # 假设浮点数在 0-1 之间，如果是 0-255 则需要先判断最大值
                    # 这里做一个简单的启发式判断：如果最大值 > 1.0，则不乘 255
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            # 4. 处理维度 (针对非 1D 输入，或 1D 解析后的后续处理)
            if image.ndim == 2:
                # 灰度图 (H, W) -> (H, W, 1)
                image = image[:, :, np.newaxis]
            elif image.ndim == 3:
                # 检查是否是 CHW 格式 (C, H, W)
                # 如果第一个维度是通道数 (1, 3, 4) 且小于第二个维度 (Height)
                if image.shape[0] in [1, 3, 4] and image.shape[0] < image.shape[1]:
                    # 转置为 HWC: (C, H, W) -> (H, W, C)
                    image = np.transpose(image, (1, 2, 0))
            
            # 5. 最终校验形状
            if image.shape[0] != height or image.shape[1] != width:
                log.warning(f"Image shape {image.shape} does not match expected height={height}, width={width}. Resizing might be needed.")
                # 可选：在这里添加 cv2.resize 逻辑，但通常直接报错或警告更好，避免静默错误

            return image

        except Exception as e:
            log.error(f"Failed to preprocess image: {e}", exc_info=True)
            return None

    def view_image(self, name: str, images: np.ndarray, timestamps: np.ndarray):
        """
        显示图像,
        
        Args:
            name: 图像名字
            images:(HWC)
            timestamps: 
        """
        log.info(f"view_image: {name}: {len(images)} 个点")

        if len(images) <= 1 or len(timestamps) <= 1:
            return
            
        # 创建轨迹实体路径
        entity_path = f"encoded_images/{name}"
        
        # 按时间顺序记录每个点
        for idx, (image, ts) in enumerate(zip(images, timestamps)):
            self._set_timestamp(ts)

            retval, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            
            if retval:
                encoded_image = buffer.tobytes()
                rr.log(
                    f"{entity_path}",
                    rr.EncodedImage(contents=encoded_image,media_type="image/jpeg")
                )

    def view_trajectory(self, name: str, positions: np.ndarray, orientations: np.ndarray, timestamps: np.ndarray):
        """
        显示轨迹, 轨迹应该在世界坐标系下，世界坐标系以初始第一帧的头部正前方为iX，重力加速度方向相反为Z，Y与XZ正交，右手系
        
        Args:
            name: 轨迹名字
            positions: 
            orientations:
            timestamps: 
        """
        log.info(f"view_trajectory: {name}: {len(positions)} 个点")

        if len(positions) <= 1 or len(timestamps) <= 1:
            return
            
        # 检查是否有方向数据
        has_orientations = len(orientations) == len(positions) and all(o is not None for o in orientations)
        
        # 获取颜色
        color = self.traj_colors.get(name, self.default_colors)

        # 创建轨迹实体路径
        entity_path = f"world/trajectories/{name}"
        
        # 绘制轨迹起点和终点
        rr.log(
            f"{entity_path}/point_start",
            rr.Points3D([positions[0]], radii=0.01, colors=[0, 255, 0]),
            static=True
        )
        rr.log(
            f"{entity_path}/point_end",
            rr.Points3D([positions[-1]], radii=0.01, colors=[255, 0, 0]),
            static=True
        )

        # 按时间顺序记录每个点
        for idx, (pos, ts) in enumerate(zip(positions, timestamps)):
            self._set_timestamp(ts)

            rr.log(
                f"{entity_path}/points",
                rr.Points3D(positions[:idx+1], radii=0.005, colors=color)
            )

            rr.log(
                f"{entity_path}/line",
                rr.LineStrips3D(positions[:idx+1], colors=color)
            )

            rr.log(
                f"{entity_path}/current_point",
                rr.Points3D([pos], radii=0.005, colors=color)
            )
            
            # 如果存在方向数据，记录当前时间的坐标系
            if has_orientations:
                qx, qy, qz, qw = orientations[idx]
                axis_length = 0.05  # 坐标系轴长度
                
                # 创建当前时间的坐标系变换
                rr.log(
                    f"{entity_path}/current_frame",
                    rr.Transform3D(
                        translation=pos,
                        rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw])
                    )
                )
                
                rr.log(
                    f"{entity_path}/current_frame/axes",
                    rr.Arrows3D(
                        origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        vectors=[
                            [axis_length, 0, 0],  # X轴
                            [0, axis_length, 0],  # Y轴
                            [0, 0, axis_length],  # Z轴
                        ],
                        colors=[
                            [255, 0, 0],  # 红色 X轴
                            [0, 255, 0],  # 绿色 Y轴
                            [0, 0, 255],  # 蓝色 Z轴
                        ]
                    )
                )

    def view_camera_frustum(self, \
            name: str, positions: np.ndarray, orientations: np.ndarray, timestamps: np.ndarray, \
            K, width, height
        ):           

        log.info(f"view_camera_frustum: 📷 {name}: {len(positions)} 个点")

        if len(positions) <= 1 or len(timestamps) <= 1:
            return

        color = self.camera_colors.get(name, self.default_colors)
        
        # 创建相机实体路径
        entity_path = f"world/cameras/{name}"
            
        
        for idx, (pos, ts) in enumerate(zip(positions, timestamps)):
            self._set_timestamp(ts)
            
            # 记录相机位置点
            rr.log(
                f"{entity_path}/position",
                rr.Points3D([pos], radii=0.002, colors=color)
            )
            
            # 如果有方向数据，记录相机坐标系
            if idx < len(orientations):
                qx, qy, qz, qw = orientations[idx]
                axis_length = 0.02  # 坐标系轴长度

                # 创建相机坐标系
                rr.log(
                    f"{entity_path}/frame",
                    rr.Transform3D(
                        translation=pos,
                        rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw])
                    )
                )

                rr.log(
                    f"{entity_path}/frame/axes",
                    rr.Arrows3D(
                        origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                        vectors=[
                            [axis_length, 0, 0],  # X轴
                            [0, axis_length, 0],  # Y轴
                            [0, 0, axis_length],  # Z轴
                        ],
                        colors=[
                            [255, 0, 0],  # 红色 X轴
                            [0, 255, 0],  # 绿色 Y轴
                            [0, 0, 255],  # 蓝色 Z轴
                        ]
                    )
                )
                
                # 在相机坐标系中记录视锥体
                create_camera_frustum(f"{entity_path}/frame", K, width, height, color)
                      
        

    def view_hand_landmarks(
        self,
        hand_landmarks_list: list,
        handedness_list: list,
        timestamps: np.ndarray,
        camera_pose: dict = None,
        scale: float = 0.1,
    ):
        """
        在 Rerun world 坐标系中可视化手部关键点。
        
        Args:
            hand_landmarks_list: 每帧的手部关键点列表，每个元素是长度为 N_hand 的列表，
                                 每个手包含 21 个关键点，每个关键点有 x, y, z 属性（归一化坐标）。
            handedness_list: 每帧的左右手信息，用于区分左右手颜色。
            timestamps: 时间戳列表（纳秒），长度与 hand_landmarks_list 相同。
            camera_pose: 可选，相机在世界坐标系中的位姿，用于将手部坐标转换到世界坐标系。
                         格式: {'position': [x, y, z], 'orientation': [qx, qy, qz, qw]}
            scale: 手部缩放因子，用于调整手部在3D空间中的显示大小。
        """
        if len(hand_landmarks_list) == 0 or len(timestamps) == 0:
            return
        
        # 定义左右手颜色
        left_color = [100, 200, 255]   # 蓝色
        right_color = [255, 100, 100]  # 红色
        
        entity_path = "world/hands"
        
        for frame_idx, (hand_landmarks, handedness, ts) in enumerate(zip(
            hand_landmarks_list, handedness_list, timestamps
        )):
            self._set_timestamp(ts)
            
            # 遍历每只手
            for hand_idx, (hand_landmarks_single, handedness_single) in enumerate(
                zip(hand_landmarks, handedness)
            ):
                # 确定颜色
                is_left = handedness_single[0].category_name == "Left"
                color = left_color if is_left else right_color
                hand_label = "left" if is_left else "right"
                
                # 提取关键点坐标
                # MediaPipe 坐标: x, y 归一化到 [0, 1]，z 是相对深度
                # 需要将归一化坐标转换为实际3D坐标
                xs = np.array([landmark.x for landmark in hand_landmarks_single])
                ys = np.array([landmark.y for landmark in hand_landmarks_single])
                zs = np.array([landmark.z for landmark in hand_landmarks_single])
                
                # 应用缩放
                points_3d = np.stack([xs * scale, ys * scale, zs * scale], axis=-1)
                
                # 如果有相机位姿，将手部坐标转换到世界坐标系
                if camera_pose is not None:
                    cam_pos = np.array(camera_pose['position'])
                    cam_quat = np.array(camera_pose['orientation'])  # [qx, qy, qz, qw]
                    
                    # 创建旋转矩阵
                    from scipy.spatial.transform import Rotation as R
                    rot = R.from_quat(cam_quat)  # MediaPipe 使用 [x, y, z, w]
                    
                    # 旋转点
                    points_3d = rot.apply(points_3d)
                    
                    # 平移
                    points_3d = points_3d + cam_pos
                
                hand_entity_path = f"{entity_path}/{hand_label}"
                
                # 记录关键点
                rr.log(
                    f"{hand_entity_path}/points",
                    rr.Points3D(
                        positions=points_3d,
                        radii=0.0003,
                        colors=color,
                    )
                )
                
                # 绘制连接线
                for start_idx, end_idx in HAND_CONNECTIONS:
                    start_point = points_3d[start_idx]
                    end_point = points_3d[end_idx]
                    
                    rr.log(
                        f"{hand_entity_path}/line_{start_idx}_{end_idx}",
                        rr.LineStrips3D(
                            strips=[start_point, end_point],
                            colors=color,
                        )
                    )
    
    def view_hand_landmarks_from_detection(
        self,
        detection_results: list,
        timestamps: np.ndarray,
        camera_info: dict = None,
        scale: float = 0.1,
    ):
        """
        从 MediaPipe 检测结果可视化手部关键点。
        
        Args:
            detection_results: 每帧的 MediaPipe 检测结果，包含 hand_landmarks 和 handedness。
            timestamps: 时间戳列表（纳秒）。
            camera_info: 可选，相机信息，包含内参和位姿。
            scale: 手部缩放因子。
        """
        hand_landmarks_list = []
        handedness_list = []
        
        for result in detection_results:
            if result is not None and result.hand_landmarks:
                hand_landmarks_list.append(result.hand_landmarks)
                handedness_list.append(result.handedness)
            else:
                # 跳过没有检测到手部的帧
                hand_landmarks_list.append([])
                handedness_list.append([])
        
        # 过滤掉空帧
        valid_indices = [i for i, h in enumerate(hand_landmarks_list) if len(h) > 0]
        if valid_indices:
            hand_landmarks_list = [hand_landmarks_list[i] for i in valid_indices]
            handedness_list = [handedness_list[i] for i in valid_indices]
            timestamps = [timestamps[i] for i in valid_indices]
            
            self.view_hand_landmarks(hand_landmarks_list, handedness_list, timestamps, camera_info, scale)

    def view_depth_maps(self, name: str, depth_maps: np.ndarray, timestamps: np.ndarray):
        """
        显示深度图序列
        
        Args:
            name: 深度图名称
            depth_maps: 深度图列表，每个元素是 (H, W) 的 numpy 数组
            timestamps: 时间戳列表（纳秒）
        """
        log.info(f"view_depth_maps: {name}: {len(depth_maps)} 帧")

        log.info(f"depth_maps.shape: {depth_maps.shape}")
        
        print(f"depth_maps[0]: {depth_maps[0]}")
        if len(depth_maps) <= 1 or len(timestamps) <= 1:
            return

        entity_path = f"depth/{name}"

        for idx, (depth, ts) in enumerate(zip(depth_maps, timestamps)):
            self._set_timestamp(ts)

            # 将深度图转换为可视化图像（归一化到 0-255）
            # 深度范围 0-5m
            max_depth = 10.0
            depth_normalized = np.clip(depth, 0, max_depth) / max_depth
            depth_colormap = cv2.applyColorMap(
                (depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            depth_colormap[depth == 0] = [0, 0, 0]  # 无效深度设为黑色

            rr.log(
                f"{entity_path}",
                rr.Image(depth_colormap)
            )

    # def view_camera_data(self, )
