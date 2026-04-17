from pathlib import Path
import argparse
import numpy as np

from roboegopipe.dataloader.genrobot import GenrobotdataLoader
from roboegopipe.viewer.viewer import Viewer
from roboegopipe.viewer.camera import create_camera_frustum, compute_camera_world_pose

import logging
from rich.logging import RichHandler


logging.basicConfig(
    level=logging.INFO,                    # 设置日志级别
    format="%(message)s",                  # 只显示消息本身
    datefmt="[%X]",                        # 时间格式
    handlers=[RichHandler()]               # 关键：使用 RichHandler
)
log = logging.getLogger()


def _short_name(name: str, id = -1):
    return name.split('/')[id]

def parse_and_view_camera_image(viewer: Viewer, images):
    for topic, data in images.items():
        name = _short_name(topic, -2)
        topic_images = np.array(data["images"], dtype=np.float32)
        topic_timestamps = np.array(data["timestamps"], dtype=np.float64)
        viewer.view_image(name, topic_images, topic_timestamps)

def parse_and_view_traj_data(viewer: Viewer, traj):
    for topic, data in traj.items():
        name = _short_name(topic)
        positions = np.array(data["positions"], dtype=np.float32)
        orientations = np.array(data["orientations"], dtype=np.float32)
        timestamps = np.array(data["timestamps"], dtype=np.float64)
        viewer.view_trajectory(name, positions, orientations, timestamps)

def parse_and_view_camera_frustum(viewer: Viewer, camera_info):
    for topic, data in camera_info.items():
        if not data["info"]:
            continue
            
        # 相机内参信息
        print(data["info"][0])
        camera_data_init = data["info"][0]

        K = camera_data_init.get('K', [])
        width = camera_data_init.get('width', 0)
        height = camera_data_init.get('height', 0)

        log.debug(f"K参数: {K}")
        log.debug(f"分辨率: {width}x{height}")
        log.debug(f"畸变模型: {camera_data_init.get('distortion_model', 'unknown')}")
        
        # 获取相机名称和颜色
        name = _short_name(topic)
        positions = []
        orientations = []
        
        for camera_data in data["info"]:
            T_b_c = camera_data.get('T_b_c', [])
            if len(T_b_c) >= 7:
                x, y, z = T_b_c[0], T_b_c[1], T_b_c[2]
                qx, qy, qz, qw = T_b_c[3], T_b_c[4], T_b_c[5], T_b_c[6]
                positions.append([x, y, z])
                orientations.append([qx, qy, qz, qw])
        
        if not positions:
            print(f"⚠️ 相机 {name} 没有有效的位置数据")
            continue

        positions = np.array(positions, dtype=np.float32)
        orientations = np.array(orientations, dtype=np.float32)
        timestamps = np.array(data["timestamps"], dtype=np.float64)
        viewer.view_camera_frustum(name, positions, orientations, timestamps, K, width, height)

def parse_and_view_camera_move(viewer: Viewer, trajectories, camera_info):           
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

        print(f"first_camera_info: {first_camera_info}")
        T_b_c = first_camera_info.get('T_b_c', [])
        
        if len(T_b_c) < 7:
            print(f"⚠️ 相机 {camera_topic.split('/')[-2]} 没有有效的T_b_c数据")
            continue
        
        camera_name = camera_topic.split('/')[-1]
        
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

    parse_and_view_traj_data(viewer, trajectories)
    parse_and_view_camera_frustum(viewer, corrected_camera_info)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='可视化MCAP文件中的轨迹和相机数据')
    parser.add_argument('--mcap_path', type=str, 
                       default="/home/ryu-yang/Documents/Datasets/Domestic_Services/Living_Room/Organization/Organize_desktop/3a8f559dfb0847c8be710fa31c37758a.mcap",
                       help='MCAP文件路径')
    parser.add_argument('--urdf', type=str, 
                       default="descriptions/genrobot/ego_v2.urdf",
                       help='URDF文件路径')
    parser.add_argument('--mode', type=str, choices=['traj', 'camera_frustum', 'both','camera_move','camera_data'], default='traj',
                       help='可视化模式: traj(运动轨迹), camera_frustum(相机视锥体), camera_data(相机画面)')
    parser.add_argument('--time_alignment', type=str, choices=['nearest', 'linear'], default='nearest',
                       help='时间对齐方法: nearest(最近邻), linear(线性插值)')
    
    
    args = parser.parse_args()
    mcap_path = args.mcap_path
    
    # 检查文件是否存在
    if not Path(mcap_path).exists():
        log.error(f"❌ 文件不存在: {mcap_path}")
        log.error("请使用 --mcap_path 参数指定有效的MCAP文件路径")
        return
    
    log.info("🔧 初始化数据加载器...\n")
    dataLoader = GenrobotdataLoader(mcap_path)
    
    log.info("📖 读取数据...")
    dataLoader.read_data(decode_images=True)
    
    # 获取轨迹和相机数据
    traj = dataLoader.get_traj()
    camera_info = dataLoader.get_camera_info(from_urdf=True, urdf_path=args.urdf)
    
    # 解码图像（如果需要）
    log.info("🔄 解码图像数据...")
    images = dataLoader.decode_all_images()
    
    log.info("📊 数据统计:")
    log.info(f"  - 轨迹数据: {len(traj)} 个topic")
    for topic, data in traj.items():
        short_name = topic.split('/')[-1]
        log.info(f"    * {short_name}: {len(data['positions'])} 个点")
    
    log.info(f"  - 相机数据: {len(camera_info)} 个相机")
    for topic, data in camera_info.items():
        camera_name = topic.split('/')[-1]
        log.info(f"    * {camera_name}: {len(data['info'])} 条信息")
    
    log.info(f"🎯 可视化模式: {args.mode}\n")
    log.info(f"⏰ 时间对齐方法: {args.time_alignment}\n")
    


    viewer = Viewer()

    if args.mode == 'traj' and traj:
        parse_and_view_traj_data(viewer, traj)

    if args.mode == 'camera_frustum' and camera_info:
        parse_and_view_camera_frustum(viewer, camera_info)
        
    # if args.mode == 'both' and camera_info and traj:
    #     parse_and_view_traj_data(viewer, traj)
    #     parse_and_view_camera_frustum(viewer, camera_info)

    if args.mode == 'camera_move' and camera_info and traj:
        parse_and_view_camera_move(viewer, traj, camera_info)
        parse_and_view_camera_image(viewer, images)

    viewer.flush()

if __name__ == "__main__":
    main()
