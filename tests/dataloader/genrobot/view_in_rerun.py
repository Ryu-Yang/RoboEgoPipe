from pathlib import Path
import argparse

from roboegopipe.dataloader.genrobot import GenrobotdataLoader
from roboegopipe.viewer.viewer import Viewer
# from roboegopipe.viewer.camera import (
#     visualize_camera_with_rerun, 
#     visualize_trajectory_and_camera,
#     visualize_camera_with_trajectory
# )


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
        print(f"❌ 文件不存在: {mcap_path}")
        print("请使用 --mcap_path 参数指定有效的MCAP文件路径")
        return
    
    print("🔧 初始化数据加载器...")
    dataLoader = GenrobotdataLoader(mcap_path)
    
    print("📖 读取数据...")
    dataLoader.read_data()
    
    # 获取轨迹和相机数据
    traj = dataLoader.get_traj()
    camera_info = dataLoader.get_camera_info(from_urdf=True, urdf_path=args.urdf)
    
    print("\n📊 数据统计:")
    print(f"  - 轨迹数据: {len(traj)} 个topic")
    for topic, data in traj.items():
        short_name = topic.split('/')[-1]
        print(f"    * {short_name}: {len(data['positions'])} 个点")
    
    print(f"  - 相机数据: {len(camera_info)} 个相机")
    for topic, data in camera_info.items():
        camera_name = topic.split('/')[-1]
        print(f"    * {camera_name}: {len(data['info'])} 条信息")
    
    print(f"\n🎯 可视化模式: {args.mode}")
    print(f"⏰ 时间对齐方法: {args.time_alignment}")
    
    # if camera_info and traj:
    #     if args.mode == 'trajectory_only':
    #         print("\n🎨 可视化 只显示轨迹数据（不显示相机轨迹）...")
    #         visualize_trajectory_with_rerun(traj)
    #     if args.mode == 'original':
    #         print("\n🎨 可视化原始相机位置（不考虑body轨迹）...")
    #         visualize_trajectory_and_camera(traj, camera_info)
    #     elif args.mode == 'corrected':
    #         print("\n🎨 可视化校正后的相机位置（考虑body轨迹）...")
    #         visualize_camera_with_trajectory(camera_info, traj, 
    #                                         use_camera_timestamps=args.use_camera_timestamps,
    #                                         time_alignment_method=args.time_alignment)
    # else:
    #     print("❌ 没有有效数据可供可视化")

    viewer = Viewer()

    if args.mode == 'traj' and traj:
        viewer.view_trajectory(traj)
    if args.mode == 'camera_frustum' and camera_info:
        viewer.view_camera_frustum(camera_info)
    if args.mode == 'both' and camera_info and traj:
        viewer.view_trajectory(traj)
        viewer.view_camera_frustum(camera_info)
    if args.mode == 'camera_move' and camera_info and traj:
        viewer.view_camera_move(camera_info, traj)
    # if args.mode == 'camera_data' and camera_info and traj:
    #     viewer.view_camera_move(camera_info, traj)


if __name__ == "__main__":
    main()
