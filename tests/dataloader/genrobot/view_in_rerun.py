from pathlib import Path
import argparse
import numpy as np
import re

from roboegopipe.dataloader.genrobot import GenrobotdataLoader
from roboegopipe.depthestimator.stereo import StereoEstimator,creat_matrix_from_pose
from roboegopipe.viewer.viewer import Viewer
from roboegopipe.viewer.camera import create_camera_frustum, compute_camera_world_pose
from scipy.spatial.transform import Rotation as R
import logging
from rich.logging import RichHandler
from roboegopipe.mediapipe.detector import Detector

logging.basicConfig(
    level=logging.INFO,                    # 设置日志级别
    format="%(message)s",                  # 只显示消息本身
    datefmt="[%X]",                        # 时间格式
    handlers=[RichHandler()]               # 关键：使用 RichHandler
)
log = logging.getLogger()


def find_stereo_pair(camera_info, images, left_cam_idx=2, right_cam_idx=3):
    """
    查找指定的双目相机对
    
    Args:
        camera_info: 相机信息字典
        images: 图像数据字典
        left_cam_idx: 左相机编号 (默认 2)
        right_cam_idx: 右相机编号 (默认 3)
    
    Returns:
        pair: dict 包含配对信息，或 None
    """
    left_info = None
    right_info = None
    left_img_topic = None
    right_img_topic = None
    
    # 从 camera_info 中查找左/右相机信息
    for topic, data in camera_info.items():
        match = re.search(r'camera(\d+)', topic)
        if match and data.get("info"):
            cam_idx = int(match.group(1))
            if cam_idx == left_cam_idx:
                left_info = data["info"][0]
            elif cam_idx == right_cam_idx:
                right_info = data["info"][0]
    
    if left_info is None or right_info is None:
        log.warning(f"⚠️ 未找到 camera{left_cam_idx} 或 camera{right_cam_idx} 的相机信息")
        return None
    
    # 从 images 中查找对应的图像 topic
    for topic in images.keys():
        match = re.search(r'camera(\d+)', topic)
        if match:
            cam_idx = int(match.group(1))
            if cam_idx == left_cam_idx:
                left_img_topic = topic
            elif cam_idx == right_cam_idx:
                right_img_topic = topic
    
    if left_img_topic is None or right_img_topic is None:
        log.warning(f"⚠️ 未找到 camera{left_cam_idx} 或 camera{right_cam_idx} 的图像数据")
        log.warning(f"可用图像 topics: {list(images.keys())}")
        return None
    
    width = 640
    height = 480
    
    log.info(f"📷 配对双目相机: camera{left_cam_idx} (L) <-> camera{right_cam_idx} (R)")
    
    return {
        "left_img_topic": left_img_topic,
        "right_img_topic": right_img_topic,
        "left_info": left_info,
        "right_info": right_info,
        "width": width,
        "height": height,
    }


def _short_name(name: str, id = -1):
    return name.split('/')[id]

def parse_and_view_hand_detect(viewer: Viewer, images):
    detector = Detector()
    for topic, data in images.items():
        if 'camera2' in topic:
            name = _short_name(topic, -2)+"_detected"
            match_images = np.array(data["images"], dtype=np.float32)
            match_timestamps = np.array(data["timestamps"], dtype=np.float64)

            detected_images = []
            for i in range(len(match_images)):
                detected_image = detector.detect(match_images[i], match_timestamps[i])
                detected_images.append(detected_image)

            viewer.view_image(name, detected_images, match_timestamps)


def parse_and_view_depth_data(viewer: Viewer, depths):
    for topic, data in depths.items():
        name = _short_name(topic, -2)
        depth_maps = np.array(data["depth_maps"], dtype=np.float32)
        timestamps = np.array(data["timestamps"], dtype=np.float64)
        viewer.view_depth_maps(name, depth_maps, timestamps)

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
                       default="data/3a8f559dfb0847c8be710fa31c37758a.mcap",
                       help='MCAP文件路径')
    parser.add_argument('--urdf', type=str, 
                       default="descriptions/genrobot/ego_v2.urdf",
                       help='URDF文件路径')
    parser.add_argument('--mode', type=str, choices=['traj', 'camera_frustum', 'both','camera_move','camera_data','all'], default='traj',
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
    estimator = StereoEstimator()
    
    log.info("📖 读取数据...")
    dataLoader.read_data(decode_images=True)
    
    # 获取轨迹和相机数据
    traj = dataLoader.get_traj()
    camera_info = dataLoader.get_camera_info(from_urdf=True, urdf_path=args.urdf)
    
    # 解码图像（如果需要）
    log.info("🔄 解码图像数据...")
    images = dataLoader.decode_all_images()

    # 计算深度
    log.info("🔍 计算深度数据...")
    
    # 1. 查找指定的双目相机对 (camera2, camera3)
    pair = find_stereo_pair(camera_info, images, left_cam_idx=2, right_cam_idx=3)
    
    if pair is None:
        log.warning("⚠️ 未找到有效的双目相机对，跳过深度计算")
        depth_data = {}
        camera_align_data = {}
    else:
        left_img_topic = pair["left_img_topic"]
        right_img_topic = pair["right_img_topic"]
        left_info = pair["left_info"]
        right_info = pair["right_info"]
        width = pair["width"]
        height = pair["height"]
        
        # 获取图像和时间戳
        left_images = images[left_img_topic]["images"]
        right_images = images[right_img_topic]["images"]
        left_timestamps = images[left_img_topic]["timestamps"]
        right_timestamps = images[right_img_topic]["timestamps"]
        
        log.info(f"🔧 标定双目系统: {left_img_topic.split('/')[-2]} (L) <-> {right_img_topic.split('/')[-2]} (R)")
        log.debug(f"  - 图像尺寸: {width}x{height}")
        log.debug(f"  - 左相机 T_b_c: {left_info.get('T_b_c', [])[:3]}")
        
        # 标定双目系统
        estimator.calibrate_from_camera_info(left_info, right_info, (width, height))
        
        # 批量计算深度
        pair_depth = estimator.compute_depth_batch(
            left_images, right_images, left_timestamps, right_timestamps
        )
        
        # 存储深度数据（带时间戳）
        depth_topic = "stereo/camera2_camera3"
        depth_data = {depth_topic: pair_depth}
        
        # 获取校正后的相机参数
        rect_params = estimator.get_rectified_params()
        K_rect = rect_params["K_rect"]
        R_rect = rect_params["R_rect"]
        rect_width = rect_params["width"]
        rect_height = rect_params["height"]
        
        # 计算校正后的 T_b_c
        # 校正后的相机位姿 = 原始 T_b_c * R_rect^T
        # 因为 R_rect 是校正旋转，它把相机坐标系旋转到了校正坐标系
        T_b_c_orig = left_info.get("T_b_c", [])
        T_b_c_matrix = creat_matrix_from_pose(T_b_c_orig)
        # 校正后的变换: T_b_c_rect = T_b_c * R_rect^T
        # 等价于: 先做 R_rect 旋转，再做原始的 T_b_c
        T_b_c_rect = T_b_c_matrix.copy()
        T_b_c_rect[:3, :3] = T_b_c_matrix[:3, :3] @ R_rect.T
        
        # 转换为 [x, y, z, qx, qy, qz, qw] 格式
        T_b_c_rect_list = list(T_b_c_rect[:3, 3]) + list(R.from_matrix(T_b_c_rect[:3, :3]).as_quat())
        
        log.debug(f"  - 原始 K: {left_info.get('K', [])[:3]}")
        log.debug(f"  - 校正后 K: {K_rect[0, :3]}")
        log.debug(f"  - 原始尺寸: {width}x{height} -> 校正后: {rect_width}x{rect_height}")
        
        # 构建与深度对齐的 camera_align 数据
        # camera_align 使用校正后的左相机参数，时间戳与深度一致
        camera_align_data = {
            "topic": depth_topic,
            "images": [],  # 占位，实际会在下面重新赋值
            "timestamps": pair_depth["timestamps"],
            "T_b_c": T_b_c_rect_list,
            "K": K_rect.flatten().tolist(),
            "width": rect_width,
            "height": rect_height,
        }
        
        # 按深度时间戳对齐左相机图像，并裁剪到有效区域
        align_images = []
        ts_left_array = np.array(left_timestamps)
        roi = estimator._valid_roi
        
        for ts in pair_depth["timestamps"]:
            # 找到对应的左相机图像
            ts_diff = np.abs(ts_left_array - ts)
            best_idx = np.argmin(ts_diff)
            img = left_images[best_idx]
            
            # 校正图像
            rect_img = estimator._rectify_image(img, "left")
            
            # 裁剪到有效区域 (ROI)
            if roi is not None:
                x, y, w, h = roi
                rect_img = rect_img[y:y+h, x:x+w]
            
            align_images.append(rect_img)
        
        camera_align_data["images"] = align_images
        
        log.info(f"✅ 深度计算完成: {depth_topic} ({len(pair_depth['depth_maps'])} 帧)")
        log.info(f"✅ 对齐相机数据: camera_align ({len(camera_align_data['images'])} 帧, {rect_width}x{rect_height})")
    
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

    if args.mode == 'all' and camera_info and traj and depth_data:
        parse_and_view_camera_move(viewer, traj, camera_info)
        parse_and_view_camera_image(viewer, images)
        parse_and_view_depth_data(viewer, depth_data)
        parse_and_view_hand_detect(viewer, images)

    # viewer.flush()

if __name__ == "__main__":
    main()
