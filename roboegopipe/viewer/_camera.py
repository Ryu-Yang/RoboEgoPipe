import time
import numpy as np
import rerun as rr


def create_camera_frustum(entity_path, camera_data, color):
    """
    创建相机视锥体可视化
    
    Args:
        entity_path: 实体路径
        camera_data: 相机数据字典
        color: 颜色
    """
    
    # 从相机内参矩阵K提取参数
    K = camera_data.get('K', [])
    if len(K) < 9:
        return
    
    fx = K[0]  # fx
    fy = K[4]  # fy
    cx = K[2]  # cx
    cy = K[5]  # cy
    width = camera_data.get('width', 1600)
    height = camera_data.get('height', 1300)
    
    # 计算视锥体参数
    near = 0.01  # 近平面距离 (减小)
    far = 0.02    # 远平面距离 (减小)
    
    # 计算视锥体角点（在相机坐标系中）
    # 归一化图像坐标
    top_left = np.array([(0 - cx) / fx * near, (0 - cy) / fy * near, near])
    top_right = np.array([(width - cx) / fx * near, (0 - cy) / fy * near, near])
    bottom_left = np.array([(0 - cx) / fx * near, (height - cy) / fy * near, near])
    bottom_right = np.array([(width - cx) / fx * near, (height - cy) / fy * near, near])
    
    top_left_far = top_left * (far / near)
    top_right_far = top_right * (far / near)
    bottom_left_far = bottom_left * (far / near)
    bottom_right_far = bottom_right * (far / near)
    
    # 创建视锥体线框
    corners = [
        top_left, top_right, bottom_right, bottom_left,  # 近平面
        top_left_far, top_right_far, bottom_right_far, bottom_left_far  # 远平面
    ]
    
    # 创建线框的线带：每个线段作为一个单独的线带
    line_strips = []
    for line in [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 近平面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 远平面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 连接线
    ]:
        line_strips.append([corners[line[0]], corners[line[1]]])
    
    # 记录视锥体
    rr.log(
        f"{entity_path}/frustum",
        rr.LineStrips3D(line_strips, colors=color)
    )
    
    # 记录视锥体填充面（半透明）
    frustum_color = list(color) + [64]  # 添加透明度
    rr.log(
        f"{entity_path}/frustum_fill",
        rr.Mesh3D(
            vertex_positions=corners,
            triangle_indices=[
                [0, 1, 2], [0, 2, 3],  # 近平面
                [4, 5, 6], [4, 6, 7],  # 远平面
                [0, 1, 5], [0, 5, 4],  # 上侧面
                [1, 2, 6], [1, 6, 5],  # 右侧面
                [2, 3, 7], [2, 7, 6],  # 下侧面
                [3, 0, 4], [3, 4, 7],  # 左侧面
            ],
            vertex_colors=[frustum_color] * len(corners)
        )
    )

def visualize_camera_with_rerun(camera_info, trajectories=None):
    """
    使用Rerun可视化相机信息
    
    Args:
        camera_info: 相机信息字典，格式为 {topic: {"info": [camera_data], "timestamps": [timestamp]}}
        trajectories: 可选的轨迹数据，用于显示相机与轨迹的相对位置
    """
    
    print("📷 正在启动相机可视化...")
    
    # 初始化Rerun
    rr.init("Camera Viewer", spawn=True)
    
    # 设置时间轴
    rr.set_time("timestamp", timestamp=0)
    
    # 创建3D视图
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # 计算全局最小时间（所有数据使用相同的时间基准）
    global_min_time = None
    
    # 检查轨迹的时间
    if trajectories:
        for data in trajectories.values():
            if data["timestamps"]:
                timestamps_seconds = np.array(data["timestamps"]) / 1e9
                min_time = np.min(timestamps_seconds)
                if global_min_time is None or min_time < global_min_time:
                    global_min_time = min_time
    
    # 检查相机的时间
    for topic, data in camera_info.items():
        if data["timestamps"]:
            timestamps_seconds = np.array(data["timestamps"]) / 1e9
            min_time = np.min(timestamps_seconds)
            if global_min_time is None or min_time < global_min_time:
                global_min_time = min_time
    
    if global_min_time is None:
        global_min_time = 0
    
    print(f"⏰ 时间基准: 相对时间从 {global_min_time:.3f} 秒开始")
    
    # 为每个相机创建可视化
    camera_colors = {
        "camera0": [255, 100, 100],  # 红色
        "camera1": [100, 200, 255],  # 蓝色
        "camera2": [100, 255, 150],  # 绿色
        "camera3": [255, 200, 100],  # 橙色
    }
    
    default_colors = [
        [255, 100, 100],  # 红色
        [100, 200, 255],  # 蓝色  
        [100, 255, 150],  # 绿色
        [255, 200, 100],  # 橙色
        [200, 100, 255],  # 紫色
    ]
    
    # 可视化轨迹（如果提供）
    if trajectories:
        print("📊 可视化轨迹数据...")
        for i, (topic, data) in enumerate(trajectories.items()):
            if not data["positions"]:
                continue
                
            positions = np.array(data["positions"], dtype=np.float32)
            timestamps = np.array(data["timestamps"], dtype=np.float64)
            
            # 获取颜色
            color = default_colors[i % len(default_colors)]
            
            # 简化topic名称用于显示
            short_name = topic.split('/')[-1].replace('_', ' ').title()
            short_name_safe = short_name.replace(' ', '_')
            
            # 创建轨迹实体路径
            entity_path = f"world/trajectories/{short_name_safe}"
            
            # 记录轨迹线
            rr.log(
                f"{entity_path}/line",
                rr.LineStrips3D([positions], colors=color),
                static=True
            )
            
            # 记录轨迹点（带时间戳）
            if len(timestamps) > 0:
                # 转换为秒并计算相对时间
                timestamps_seconds = timestamps / 1e9
                timestamps_relative = timestamps_seconds - global_min_time
                
                # 按时间顺序记录每个点
                for idx, (pos, ts_relative) in enumerate(zip(positions, timestamps_relative)):
                    rr.set_time("timestamp", timestamp=ts_relative)
                    rr.log(
                        f"{entity_path}/points",
                        rr.Points3D([pos], radii=0.01, colors=color)
                    )
    
    # 可视化相机
    print("📷 可视化相机信息...")
    for i, (topic, data) in enumerate(camera_info.items()):
        if not data["info"]:
            continue
            
        camera_data_list = data["info"]
        timestamps = np.array(data["timestamps"], dtype=np.float64)
        
        # 获取相机名称和颜色
        camera_name = topic.split('/')[-2]  # 例如 camera0
        color = camera_colors.get(camera_name, default_colors[i % len(default_colors)])
        
        # 创建相机实体路径
        entity_path = f"world/cameras/{camera_name}"
        
        # 提取相机位置（从T_b_c转换矩阵）
        camera_positions = []
        camera_orientations = []
        
        for camera_data in camera_data_list:
            T_b_c = camera_data.get('T_b_c', [])
            if len(T_b_c) >= 7:
                # T_b_c格式: [x, y, z, qx, qy, qz, qw]
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
            timestamps_seconds = timestamps / 1e9
            timestamps_relative = timestamps_seconds - global_min_time
            
            for idx, (pos, ts_relative) in enumerate(zip(camera_positions, timestamps_relative)):
                rr.set_time("timestamp", timestamp=ts_relative)
                
                # 记录相机位置点
                rr.log(
                    f"{entity_path}/position",
                    rr.Points3D([pos], radii=0.02, colors=color)
                )
                
                # 如果有方向数据，记录相机坐标系
                if idx < len(camera_orientations):
                    qx, qy, qz, qw = camera_orientations[idx]
                    
                    # 创建相机坐标系
                    rr.log(
                        f"{entity_path}/frame",
                        rr.Transform3D(
                            translation=pos,
                            rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw])
                        )
                    )
                    
                    # 在相机坐标系中记录视锥体
                    create_camera_frustum(f"{entity_path}/frame", camera_data_list[idx], color)
        
        # 添加相机文本标签
        rr.log(
            f"{entity_path}/label",
            rr.TextDocument(f"Camera: {camera_name}\nFrames: {len(camera_positions)}")
        )
        
        print(f"  📷 {camera_name}: {len(camera_positions)} 个位置点")
        
        # 显示相机内参信息
        if camera_data_list:
            first_camera = camera_data_list[0]
            print(f"    - 分辨率: {first_camera.get('width', 0)}x{first_camera.get('height', 0)}")
            print(f"    - 畸变模型: {first_camera.get('distortion_model', 'unknown')}")
    
    # 添加坐标系轴
    rr.log("world/axes", rr.Arrows3D(
        origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    ), static=True)
    
    print("🚀 相机可视化已启动。")
    print("💡 提示:")
    print("  - 使用鼠标拖拽旋转视图")
    print("  - 使用滚轮缩放")
    print("  - 使用时间轴控件查看相机随时间变化")
    print("  - 在Rerun界面中可切换显示/隐藏不同相机")
    
    # 保持程序运行，让用户查看可视化
    try:
        print("⏳ 可视化窗口已打开，按 Ctrl+C 退出...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 退出可视化。")




def visualize_trajectory_and_camera(trajectories, camera_info):
    """
    同时可视化轨迹和相机
    
    Args:
        trajectories: 轨迹数据
        camera_info: 相机信息
    """
    visualize_camera_with_rerun(camera_info, trajectories)


def align_timestamps(camera_timestamps, trajectory_timestamps, trajectory_data, method='nearest'):
    """
    对齐相机时间戳和轨迹时间戳
    
    Args:
        camera_timestamps: 相机时间戳数组 (纳秒)
        trajectory_timestamps: 轨迹时间戳数组 (纳秒)
        trajectory_data: 轨迹数据字典，包含positions和orientations
        method: 对齐方法，'nearest'（最近邻）或 'linear'（线性插值）
    
    Returns:
        aligned_trajectory_data: 对齐后的轨迹数据列表，每个元素对应一个相机时间戳
    """
    import numpy as np
    
    if not trajectory_timestamps or not camera_timestamps:
        return []
    
    camera_timestamps_np = np.array(camera_timestamps, dtype=np.float64)
    trajectory_timestamps_np = np.array(trajectory_timestamps, dtype=np.float64)
    
    positions = np.array(trajectory_data.get('positions', []), dtype=np.float32)
    orientations = trajectory_data.get('orientations', [])
    
    if len(positions) == 0:
        return []
    
    aligned_data = []
    
    if method == 'nearest':
        # 最近邻匹配
        for cam_ts in camera_timestamps_np:
            # 找到最接近的轨迹时间戳索引
            idx = np.argmin(np.abs(trajectory_timestamps_np - cam_ts))
            
            if idx < len(positions):
                pos = positions[idx]
                orient = orientations[idx] if idx < len(orientations) else None
                aligned_data.append({
                    'position': pos,
                    'orientation': orient,
                    'timestamp': cam_ts,
                    'traj_timestamp': trajectory_timestamps_np[idx]
                })
    
    elif method == 'linear':
        # 线性插值（仅位置，方向使用球面线性插值）
        from scipy.spatial.transform import Slerp
        from scipy.spatial.transform import Rotation as R
        
        # 确保有足够的数据点进行插值
        if len(positions) < 2:
            # 数据不足，使用最近邻
            return align_timestamps(camera_timestamps, trajectory_timestamps, trajectory_data, method='nearest')
        
        # 位置线性插值
        for cam_ts in camera_timestamps_np:
            # 找到插入位置
            idx = np.searchsorted(trajectory_timestamps_np, cam_ts)
            
            if idx == 0:
                # 在第一个点之前，使用第一个点
                pos = positions[0]
                orient = orientations[0] if orientations else None
            elif idx == len(trajectory_timestamps_np):
                # 在最后一个点之后，使用最后一个点
                pos = positions[-1]
                orient = orientations[-1] if orientations else None
            else:
                # 在两个点之间，进行插值
                t_prev = trajectory_timestamps_np[idx-1]
                t_next = trajectory_timestamps_np[idx]
                alpha = (cam_ts - t_prev) / (t_next - t_prev)
                
                # 位置线性插值
                pos_prev = positions[idx-1]
                pos_next = positions[idx]
                pos = pos_prev + alpha * (pos_next - pos_prev)
                
                # 方向球面线性插值（如果有方向数据）
                if orientations and idx-1 < len(orientations) and idx < len(orientations):
                    orient_prev = orientations[idx-1]
                    orient_next = orientations[idx]
                    if orient_prev is not None and orient_next is not None:
                        # 创建旋转对象并进行SLERP
                        rot_prev = R.from_quat([orient_prev[0], orient_prev[1], orient_prev[2], orient_prev[3]])
                        rot_next = R.from_quat([orient_next[0], orient_next[1], orient_next[2], orient_next[3]])
                        
                        # 创建SLERP插值器
                        rotations = R.concatenate([rot_prev, rot_next])
                        slerp = Slerp([0, 1], rotations)
                        rot_interp = slerp([alpha])
                        orient = rot_interp.as_quat()[0].tolist()
                    else:
                        orient = None
                else:
                    orient = None
            
            aligned_data.append({
                'position': pos,
                'orientation': orient,
                'timestamp': cam_ts,
                'traj_timestamp': cam_ts  # 使用插值后的时间
            })
    
    return aligned_data


def compute_camera_world_pose(body_pose, T_b_c):
    """
    计算相机在世界坐标系中的位姿
    
    Args:
        body_pose: 字典，包含'position'和'orientation'
                  position: [x, y, z]
                  orientation: [qx, qy, qz, qw] 或 None
        T_b_c: 相机相对于body的变换 [x, y, z, qx, qy, qz, qw]
    
    Returns:
        camera_pose: 字典，包含'position'和'orientation'
    """
    import numpy as np
    
    if len(T_b_c) < 7:
        # T_b_c数据不完整，返回None
        return None
    
    # 提取body位置和方向
    body_pos = np.array(body_pose['position'], dtype=np.float32)
    body_orient = body_pose['orientation']
    
    # 提取相机相对于body的变换
    T_b_c_pos = np.array(T_b_c[:3], dtype=np.float32)
    T_b_c_orient = np.array(T_b_c[3:7], dtype=np.float32)
    
    if body_orient is None:
        # 如果没有body方向，假设body方向为单位四元数
        body_orient = [0, 0, 0, 1]
    
    # 计算相机在世界坐标系中的位置
    # 需要将相机相对位置旋转到世界坐标系
    from scipy.spatial.transform import Rotation as R
    
    # 创建body旋转
    body_rot = R.from_quat([body_orient[0], body_orient[1], body_orient[2], body_orient[3]])
    
    # 将相机相对位置旋转到世界坐标系
    camera_pos_relative = T_b_c_pos
    camera_pos_world = body_pos + body_rot.apply(camera_pos_relative)
    
    # 计算相机在世界坐标系中的方向
    # 相机方向 = body方向 × 相机相对方向
    camera_rot_relative = R.from_quat([T_b_c_orient[0], T_b_c_orient[1], T_b_c_orient[2], T_b_c_orient[3]])
    camera_rot_world = body_rot * camera_rot_relative
    camera_orient_world = camera_rot_world.as_quat()  # [qx, qy, qz, qw]
    
    return {
        'position': camera_pos_world.tolist(),
        'orientation': camera_orient_world.tolist()
    }


def visualize_camera_with_trajectory(camera_info, trajectories, use_camera_timestamps=False, time_alignment_method='nearest'):
    """
    可视化相机，考虑body轨迹的影响
    
    Args:
        camera_info: 相机信息字典
        trajectories: 轨迹数据字典
        use_camera_timestamps: 是否使用相机内参的时间戳进行对齐（False表示相机随整个轨迹运动）
        time_alignment_method: 时间对齐方法，'nearest' 或 'linear'（仅当use_camera_timestamps=True时有效）
    """
    print("🎬 正在计算相机在世界坐标系中的轨迹...")
    
    # 获取第一个轨迹作为body轨迹（假设所有相机共享同一个body）
    body_trajectory = None
    for topic, data in trajectories.items():
        if data.get('positions'):
            body_trajectory = data
            print(f"📊 使用轨迹: {topic.split('/')[-1]} 作为body轨迹")
            break
    
    if body_trajectory is None:
        print("⚠️ 未找到有效的body轨迹，使用原始相机位置")
        visualize_camera_with_rerun(camera_info, trajectories)
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
        
        if use_camera_timestamps and camera_timestamps:
            # 使用相机内参的时间戳进行对齐（旧方法）
            print(f"  ⏰ 相机 {camera_name}: 使用相机内参时间戳进行对齐")
            
            # 对齐时间戳
            aligned_body_data = align_timestamps(
                camera_timestamps,
                body_trajectory.get('timestamps', []),
                body_trajectory,
                method=time_alignment_method
            )
            
            if not aligned_body_data:
                print(f"  ⚠️ 无法对齐相机 {camera_name} 的时间戳")
                continue
            
            # 计算校正后的相机位姿（每个相机时间戳对应一个位姿）
            corrected_info = []
            corrected_timestamps = []
            
            for i, (cam_info, aligned_body) in enumerate(zip(camera_info_list, aligned_body_data)):
                if i >= len(camera_timestamps):
                    break
                    
                T_b_c_current = cam_info.get('T_b_c', T_b_c)
                if len(T_b_c_current) >= 7:
                    camera_pose = compute_camera_world_pose(aligned_body, T_b_c_current)
                    if camera_pose:
                        corrected_cam_info = cam_info.copy()
                        corrected_cam_info['position_world'] = camera_pose['position']
                        corrected_cam_info['orientation_world'] = camera_pose['orientation']
                        corrected_cam_info['T_b_c_original'] = T_b_c_current.copy()
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
                        corrected_timestamps.append(camera_timestamps[i])
            
            if corrected_info:
                corrected_camera_info[camera_topic] = {
                    'info': corrected_info,
                    'timestamps': corrected_timestamps
                }
                print(f"  ✅ 相机 {camera_name}: 校正了 {len(corrected_info)} 个位姿（基于相机时间戳）")
        
        else:
            # 新方法：相机随整个轨迹运动（不使用相机内参的时间戳）
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
    
    if corrected_camera_info:
        print("🎨 使用校正后的相机位姿进行可视化...")
        # 使用校正后的相机信息进行可视化
        visualize_camera_with_rerun(corrected_camera_info, trajectories)
    else:
        print("⚠️ 无法校正任何相机位姿，使用原始可视化")
        visualize_camera_with_rerun(camera_info, trajectories)
