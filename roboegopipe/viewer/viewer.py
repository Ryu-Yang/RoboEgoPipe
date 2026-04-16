import time
import logging
import rerun as rr
import numpy as np

from roboegopipe.viewer.camera import create_camera_frustum, compute_camera_world_pose

log = logging.getLogger()

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
                      
        

    # def view_camera_data(self, )