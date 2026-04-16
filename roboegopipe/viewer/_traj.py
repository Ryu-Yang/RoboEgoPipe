import time

import rerun as rr
import numpy as np


def visualize_trajectory_with_rerun(trajectories):
    """
    使用Rerun可视化MCAP文件中的轨迹
    
    Args:
        mcap_file_path: MCAP文件路径
        topics_filter: 要可视化的topic列表，默认为None时使用默认topic
    """

    # ================= Rerun 可视化 =================
    print("🎨 正在启动 Rerun 可视化...")
    
    # 初始化Rerun
    rr.init("MCAP Trajectory Viewer", spawn=True)
    
    # 设置时间轴
    rr.set_time("timestamp", timestamp=0)
    
    # 为每个轨迹创建不同的颜色
    colors = {
        "/robot0/vio/eef_pose": [255, 100, 100],  # 红色
        "/robot0/vio/relative_eef_pose": [100, 200, 255],  # 蓝色
    }
    
    # 默认颜色（如果topic不在预设中）
    default_colors = [
        [255, 100, 100],  # 红色
        [100, 200, 255],  # 蓝色  
        [100, 255, 150],  # 绿色
        [255, 200, 100],  # 橙色
        [200, 100, 255],  # 紫色
    ]
    
    # 计算全局最小时间（所有轨迹使用相同的时间基准）
    global_min_time = None
    for data in trajectories.values():
        if data["timestamps"]:
            timestamps_seconds = np.array(data["timestamps"]) / 1e9
            min_time = np.min(timestamps_seconds)
            if global_min_time is None or min_time < global_min_time:
                global_min_time = min_time
    
    if global_min_time is None:
        global_min_time = 0
    
    print(f"⏰ 时间基准: 相对时间从 {global_min_time:.3f} 秒开始")
    
    # 创建3D视图
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # 为每个轨迹添加坐标系
    for i, (topic, data) in enumerate(trajectories.items()):
        if not data["positions"]:
            continue
            
        positions = np.array(data["positions"], dtype=np.float32)
        timestamps = np.array(data["timestamps"], dtype=np.float64)
        
        # 获取颜色
        color = colors.get(topic, default_colors[i % len(default_colors)])
        
        # 简化topic名称用于显示 (移除空格以避免Rerun警告)
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
            # 转换为秒并计算相对时间（使用全局时间基准）
            timestamps_seconds = timestamps / 1e9
            timestamps_relative = timestamps_seconds - global_min_time
            
            # 检查是否有方向数据
            orientations = data.get("orientations", [])
            has_orientations = len(orientations) == len(positions) and all(o is not None for o in orientations)
            
            # 按时间顺序记录每个点
            for idx, (pos, ts_relative) in enumerate(zip(positions, timestamps_relative)):
                rr.set_time("timestamp", timestamp=ts_relative)
                rr.log(
                    f"{entity_path}/points",
                    rr.Points3D([pos], radii=0.01, colors=color)
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
                    
                    # 在变换后的坐标系中记录轴（只在当前时间显示）
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
                
                # 每100个点记录一次，避免过多数据点
                if idx % 100 == 0:
                    rr.log(
                        f"{entity_path}/points_batch",
                        rr.Points3D(positions[:idx+1], radii=0.005, colors=color)
                    )
        
        # 记录起点和终点（使用相对时间0）
        rr.set_time("timestamp", timestamp=0)
        rr.log(
            f"{entity_path}/start",
            rr.Points3D([positions[0]], radii=0.01, colors=[0, 255, 0])  # 绿色起点
        )
        
        if len(positions) > 1:
            rr.log(
                f"{entity_path}/end",
                rr.Points3D([positions[-1]], radii=0.01, colors=[255, 0, 0])  # 红色终点
            )
        
        # 添加文本标签
        rr.log(
            f"{entity_path}/label",
            rr.TextDocument(f"{short_name}\nPoints: {len(positions)}")
        )
        
        print(f"  📊 {short_name}: {len(positions)} 个点")
    
    # 添加坐标系轴
    rr.log("world/axes", rr.Arrows3D(
        origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    ), static=True)
    
    # 添加坐标系标签
    rr.log("world/x_axis", rr.TextDocument("X"), static=True)
    rr.log("world/y_axis", rr.TextDocument("Y"), static=True)
    rr.log("world/z_axis", rr.TextDocument("Z"), static=True)
    
    print("🚀 Rerun可视化已启动。")
    print("💡 提示:")
    print("  - 使用鼠标拖拽旋转视图")
    print("  - 使用滚轮缩放")
    print("  - 使用时间轴控件查看轨迹随时间变化")
    print("  - 在Rerun界面中可切换显示/隐藏不同轨迹")
    
    # 保持程序运行，让用户查看可视化
    try:
        print("⏳ 可视化窗口已打开，按 Ctrl+C 退出...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 退出可视化。")