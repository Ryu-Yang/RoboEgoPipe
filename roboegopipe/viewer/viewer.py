import time

import rerun as rr
import numpy as np

from roboegopipe.viewer.camera import create_camera_frustum, compute_camera_world_pose

class Viewer():
    def __init__(self):
        rr.init("RoboEgoPipe Viewer", spawn=True)
            
        # 为每个相机创建可视化
        self.camera_colors = {
            "camera0": [255, 100, 100],  # 红色
            "camera1": [100, 200, 255],  # 蓝色
            "camera2": [100, 255, 150],  # 绿色
            "camera3": [255, 200, 100],  # 橙色
        }
        
        self.default_colors = [
            [255, 100, 100],  # 红色
            [100, 200, 255],  # 蓝色  
            [100, 255, 150],  # 绿色
            [255, 200, 100],  # 橙色
            [200, 100, 255],  # 紫色
        ]

        self.colors = {
            "/robot0/vio/eef_pose": [255, 100, 100],  # 红色
            "/robot0/vio/relative_eef_pose": [100, 200, 255],  # 蓝色
        }# 为每个轨迹创建不同的颜色

        # 创建3D视图
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # 添加坐标系轴
        rr.log("world/axes", rr.Arrows3D(
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        ), static=True)

        print("🚀 Rerun可视化已启动。")
        print("💡 提示:")
        print("  - 使用鼠标拖拽旋转视图")
        print("  - 使用滚轮缩放")
        print("  - 使用时间轴控件查看轨迹随时间变化")
        print("  - 在Rerun界面中可切换显示/隐藏不同轨迹")

    def _set_timestamp(self, timestamp_ns):
        """
        设置Rerun的时间戳
        
        Args:
            timestamp_ns: 纳秒时间戳
        """
        timestamp_seconds = timestamp_ns / 1e9
        
        rr.set_time("timestamp", timestamp=timestamp_seconds)

    def _short_name(self, name: str):
        short_name = name.split('/')[-1].replace('_', ' ').title()
        short_name_safe = short_name.replace(' ', '_')
        
        return short_name_safe

    def view_trajectory(self, trajectories):
        """
        显示轨迹
        
        Args:
            trajectories: 每个话题需要2个以上的轨迹点
        """
        # 为每个轨迹添加坐标系
        for i, (topic, data) in enumerate(trajectories.items()):
            if not data["positions"] or len(data["positions"]) <= 1 or len(data["timestamps"]) <= 1:
                continue
                
            positions = np.array(data["positions"], dtype=np.float32)
            orientations = np.array(data["orientations"], dtype=np.float32)
            timestamps = np.array(data["timestamps"], dtype=np.float64)

            # 检查是否有方向数据
            has_orientations = len(orientations) == len(positions) and all(o is not None for o in orientations)
            
            # 获取颜色
            color = self.colors.get(topic, self.default_colors[i % len(self.default_colors)])
            
            short_name = self._short_name(topic)

            print(f"  📊 {short_name}: {len(positions)} 个点")
            
            # 创建轨迹实体路径
            entity_path = f"world/trajectories/{short_name}"
            
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

    def view_camera_frustum(self, camera_info):           
        print("📷 可视化相机信息...")
        for i, (topic, data) in enumerate(camera_info.items()):
            if not data["info"]:
                continue
                
            camera_data_list = data["info"]
            timestamps = np.array(data["timestamps"], dtype=np.float64)
            
            # 获取相机名称和颜色
            camera_name = topic.split('/')[-2]  # 例如 camera0
            color = self.camera_colors.get(camera_name, self.default_colors[i % len(self.default_colors)])
            
            # 创建相机实体路径
            entity_path = f"world/cameras/{camera_name}"
            
            # 提取相机位置（从T_b_c转换矩阵）
            camera_positions = []
            camera_orientations = []
            
            for camera_data in camera_data_list:
                T_b_c = camera_data.get('T_b_c', [])
                if len(T_b_c) >= 7:
                    x, y, z = T_b_c[0], T_b_c[1], T_b_c[2]
                    qx, qy, qz, qw = T_b_c[3], T_b_c[4], T_b_c[5], T_b_c[6]
                    camera_positions.append([x, y, z])
                    camera_orientations.append([qx, qy, qz, qw])
            
            if not camera_positions:
                print(f"⚠️ 相机 {camera_name} 没有有效的位置数据")
                continue
            
            camera_positions = np.array(camera_positions, dtype=np.float32)
            
            # 记录相机位置点（带时间戳）
            if len(timestamps) > 0 and len(camera_positions) == len(timestamps):
                
                for idx, (pos, ts) in enumerate(zip(camera_positions, timestamps)):
                    self._set_timestamp(ts)
                    
                    # 记录相机位置点
                    rr.log(
                        f"{entity_path}/position",
                        rr.Points3D([pos], radii=0.005, colors=color)
                    )
                    
                    # 如果有方向数据，记录相机坐标系
                    if idx < len(camera_orientations):
                        qx, qy, qz, qw = camera_orientations[idx]
                        axis_length = 0.05  # 坐标系轴长度

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
                        create_camera_frustum(f"{entity_path}/frame", camera_data_list[idx], color)
            
            print(f"  📷 {camera_name}: {len(camera_positions)} 个位置点")
            
            # 显示相机内参信息
            if camera_data_list:
                first_camera = camera_data_list[0]
                print(f"    - 分辨率: {first_camera.get('width', 0)}x{first_camera.get('height', 0)}")
                print(f"    - 畸变模型: {first_camera.get('distortion_model', 'unknown')}")
                
        
    def view_camera_move(self, camera_info, trajectories):           
        print("🎬 正在计算相机在世界坐标系中的轨迹...")
        
        # 获取世界坐标系轨迹作为body轨迹（假设所有相机共享同一个body）
        body_trajectory = None
        for topic, data in trajectories.items():
            if data.get('positions'):
                body_trajectory = data
                print(f"📊 使用轨迹: {topic.split('/')[-1]} 作为body轨迹")
                break

        if body_trajectory is None:
            print("⚠️ 未找到有效的body轨迹，使用原始相机位置")
            return

        # 为每个相机计算校正后的位姿
        corrected_camera_info = {}
        
        for camera_topic, camera_data in camera_info.items():
            camera_timestamps = camera_data.get('timestamps', [])
            camera_info_list = camera_data.get('info', [])
            
            if not camera_info_list:
                continue
            
            # 获取相机的T_b_c（使用第一个相机信息，假设所有时间点相同）
            # 注意：相机内参通常是静态或变化很慢的
            first_camera_info = camera_info_list[0]
            T_b_c = first_camera_info.get('T_b_c', [])
            
            if len(T_b_c) < 7:
                print(f"⚠️ 相机 {camera_topic.split('/')[-2]} 没有有效的T_b_c数据")
                continue
            
            camera_name = camera_topic.split('/')[-2]
            
            # 新方法：相机随整个轨迹运动
            print(f"  📍 相机 {camera_name}: 随整个轨迹运动（{len(body_trajectory['positions'])} 个点）")
            
            # 为轨迹上的每个点计算相机位姿
            corrected_info = []
            corrected_timestamps = []
            
            positions = body_trajectory.get('positions', [])
            orientations = body_trajectory.get('orientations', [])
            traj_timestamps = body_trajectory.get('timestamps', [])
            
            for i in range(len(positions)):
                # 创建body位姿
                body_pose = {
                    'position': positions[i],
                    'orientation': orientations[i] if i < len(orientations) else None
                }
                
                # 计算相机在世界坐标系中的位姿
                camera_pose = compute_camera_world_pose(body_pose, T_b_c)
                if camera_pose:
                    # 创建校正后的相机信息
                    corrected_cam_info = first_camera_info.copy()
                    corrected_cam_info['position_world'] = camera_pose['position']
                    corrected_cam_info['orientation_world'] = camera_pose['orientation']
                    corrected_cam_info['T_b_c_original'] = T_b_c.copy()
                    corrected_cam_info['T_b_c'] = [
                        camera_pose['position'][0],
                        camera_pose['position'][1],
                        camera_pose['position'][2],
                        camera_pose['orientation'][0],
                        camera_pose['orientation'][1],
                        camera_pose['orientation'][2],
                        camera_pose['orientation'][3]
                    ]
                    
                    corrected_info.append(corrected_cam_info)
                    # 使用轨迹的时间戳
                    corrected_timestamps.append(traj_timestamps[i] if i < len(traj_timestamps) else 0)
            
            if corrected_info:
                corrected_camera_info[camera_topic] = {
                    'info': corrected_info,
                    'timestamps': corrected_timestamps
                }
                print(f"  ✅ 相机 {camera_name}: 生成了 {len(corrected_info)} 个轨迹点上的相机位姿")

        self.view_trajectory(trajectories)
        self.view_camera_frustum(corrected_camera_info)