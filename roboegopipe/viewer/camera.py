import time
import numpy as np
import rerun as rr


def create_camera_frustum(entity_path, K, width, height, color, near=0.01, far=0.02):
    """
    创建相机视锥体可视化
    
    Args:
        entity_path: rerun实体路径
        k: 相机内参k
        width: 相机宽度
        height: 相机高度
        color: 颜色
        near: 视锥体参数-近平面距离
        far: 视锥体参数-远平面距离
    """
    # print(entity_path, K, width, height, color, near, far)
    
    if len(K) < 9:
        return
    
    fx = K[0]  # fx
    fy = K[4]  # fy
    cx = K[2]  # cx
    cy = K[5]  # cy
    
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
