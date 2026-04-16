#!/usr/bin/env python3
"""
URDF TF可视化脚本
用于查看URDF文件中的所有tf（transform frames）关系
支持热重载（hot reload）功能
"""

import time
import numpy as np
import rerun as rr
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from roboegopipe.dataloader.utils import (parse_urdf, rpy_to_matrix, xyz_rpy_to_matrix, 
    compute_tf_tree
)
# def parse_urdf(urdf_path):
#     """
#     解析URDF文件，提取所有link和joint的tf关系
    
#     Args:
#         urdf_path: URDF文件路径
        
#     Returns:
#         links: 字典，link名称 -> link信息
#         joints: 字典，joint名称 -> joint信息（包含parent, child, origin变换）
#     """
#     tree = ET.parse(urdf_path)
#     root = tree.getroot()
    
#     links = {}
#     joints = {}
    
#     # 解析所有link
#     for link_elem in root.findall('link'):
#         link_name = link_elem.get('name')
#         links[link_name] = {
#             'name': link_name,
#             'inertial': None,
#             'visual': None,
#             'collision': None
#         }
        
#         # 解析惯性参数
#         inertial_elem = link_elem.find('inertial')
#         if inertial_elem is not None:
#             origin_elem = inertial_elem.find('origin')
#             if origin_elem is not None:
#                 xyz = origin_elem.get('xyz', '0 0 0').split()
#                 rpy = origin_elem.get('rpy', '0 0 0').split()
#                 links[link_name]['inertial'] = {
#                     'xyz': [float(x) for x in xyz],
#                     'rpy': [float(r) for r in rpy]
#                 }
    
#     # 解析所有joint
#     for joint_elem in root.findall('joint'):
#         joint_name = joint_elem.get('name')
#         joint_type = joint_elem.get('type', 'fixed')
        
#         parent_elem = joint_elem.find('parent')
#         child_elem = joint_elem.find('child')
        
#         if parent_elem is None or child_elem is None:
#             continue
            
#         parent_link = parent_elem.get('link')
#         child_link = child_elem.get('link')
        
#         # 解析origin变换
#         origin_elem = joint_elem.find('origin')
#         if origin_elem is not None:
#             xyz = origin_elem.get('xyz', '0 0 0').split()
#             rpy = origin_elem.get('rpy', '0 0 0').split()
#             origin = {
#                 'xyz': [float(x) for x in xyz],
#                 'rpy': [float(r) for r in rpy]
#             }
#         else:
#             origin = {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]}
        
#         joints[joint_name] = {
#             'name': joint_name,
#             'type': joint_type,
#             'parent': parent_link,
#             'child': child_link,
#             'origin': origin
#         }
    
#     return links, joints


# def rpy_to_matrix(rpy):
#     """
#     将RPY（roll, pitch, yaw）转换为4x4变换矩阵
    
#     Args:
#         rpy: [roll, pitch, yaw] in radians
        
#     Returns:
#         4x4变换矩阵
#     """
#     rot = R.from_euler('xyz', rpy).as_matrix()
#     T = np.eye(4)
#     T[:3, :3] = rot
#     return T


# def xyz_rpy_to_matrix(xyz, rpy):
#     """
#     将位置（xyz）和方向（rpy）转换为4x4变换矩阵
    
#     Args:
#         xyz: [x, y, z] 位置
#         rpy: [roll, pitch, yaw] in radians
        
#     Returns:
#         4x4变换矩阵
#     """
#     rot = R.from_euler('xyz', rpy).as_matrix()
#     T = np.eye(4)
#     T[:3, :3] = rot
#     T[:3, 3] = xyz
#     return T


# def compute_tf_tree(joints, base_link='base_link'):
#     """
#     计算从各个link到base_link的变换矩阵（base_link_T_link）
     
#     Args:
#         joints: urdf joint字典
#         base_link: base_link名称
        
#     Returns:
#         tf_dict: 字典，link名称 -> 从该link到base_link的4x4变换矩阵（base_link_T_link）
#     """
#     if base_link is None:
#         raise ValueError("base_link is not defined or is None")
#     tf_dict = {base_link: np.eye(4)}
    
#     # 使用BFS遍历tf树
#     queue = [base_link]
#     visited = set([base_link])
    
#     while queue:
#         current_link = queue.pop(0)
        
#         # 查找所有以current_link为parent的joint
#         for _joint_name, joint_info in joints.items():
#             if joint_info['parent'] == current_link:
#                 child_link = joint_info['child']
                
#                 if child_link in visited:
#                     continue
                    
#                 # joint的origin矩阵是parent_T_child（child相对于parent的位姿）
#                 parent_T_child = xyz_rpy_to_matrix(
#                     joint_info['origin']['xyz'],
#                     joint_info['origin']['rpy']
#                 )
                
#                 # 已知parent_T_base（从base到parent的变换）
#                 base_T_parent = tf_dict[current_link]
                
#                 # 计算child_T_base = child_T_parent @ parent_T_base
#                 base_T_child = base_T_parent @  parent_T_child
                
#                 tf_dict[child_link] = base_T_child
#                 queue.append(child_link)
#                 visited.add(child_link)
    
#     return tf_dict


def visualize_tf_tree(tf_dict):
    """
    使用Rerun可视化tf树
    
    Args:
        tf_dict: tf字典，link名称 -> 变换矩阵
    """
    # 初始化Rerun
    rr.init("URDF TF Viewer", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # 添加坐标系轴
    rr.log("world/axes", rr.Arrows3D(
        origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    ), static=True)
    
    # 为每个link定义颜色
    link_colors = {}
    colors = [
        [255, 100, 100],  # 红色
        [100, 200, 255],  # 蓝色
        [100, 255, 150],  # 绿色
        [255, 200, 100],  # 橙色
        [200, 100, 255],  # 紫色
        [255, 255, 100],  # 黄色
        [100, 255, 255],  # 青色
        [255, 100, 255],  # 粉色
    ]
    
    link_names = list(tf_dict.keys())
    for i, link_name in enumerate(link_names):
        link_colors[link_name] = colors[i % len(colors)]
    
    # 可视化每个link的坐标系
    for link_name, base_T_link in tf_dict.items():
        
        # 提取位置和方向
        position = base_T_link[:3, 3]  # link原点在base_link坐标系中的位置
        rotation_matrix = base_T_link[:3, :3]  # 从base_link到link的旋转
        
        # 将旋转矩阵转换为四元数
        rot = R.from_matrix(rotation_matrix)
        quat = rot.as_quat()  # [qx, qy, qz, qw]
        
        color = link_colors[link_name]
        
        # 创建link实体路径
        entity_path = f"world/links/{link_name}"
        
        # 记录link坐标系
        rr.log(
            f"{entity_path}/frame",
            rr.Transform3D(
                translation=position,
                rotation=rr.Quaternion(xyzw=[quat[0], quat[1], quat[2], quat[3]])
            ),
            static=True
        )
        
        # 记录坐标系轴
        axis_length = 0.05
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
            ),
            static=True
        )
        
        # 记录link名称文本
        rr.log(
            f"{entity_path}/label",
            rr.TextDocument(f"{link_name}"),
            static=True
        )
        
        # 记录link位置点
        rr.log(
            f"{entity_path}/position",
            rr.Points3D([position], radii=0.01, colors=color),
            static=True
        )
    
    # 可视化joint连接线（从base_link原点到各个link原点）
    for link_name, base_T_link in tf_dict.items():
        position = base_T_link[:3, 3]  # link原点在base_link坐标系中的位置
        color = link_colors[link_name]
        
        # 为每个link添加连接线到base_link原点
        entity_path = f"world/links/{link_name}"
        
        # 记录从base_link原点到link原点的连接线
        rr.log(
            f"{entity_path}/connection",
            rr.LineStrips3D([[[0, 0, 0], position]], colors=color),
            static=True
        )
    
    print(f"✅ 可视化完成！共显示 {len(tf_dict)} 个link的tf关系")
    print("💡 提示:")
    print("  - 使用鼠标拖拽旋转视图")
    print("  - 使用滚轮缩放")
    print("  - 在Rerun界面中可以切换显示/隐藏不同的link")


def print_tf_info(tf_dict):
    """
    打印tf信息
    
    Args:
        tf_dict: tf字典，存储base_T_link（从link到base_link的变换）
    """
    print("\n📊 TF变换信息:")
    print("=" * 80)
    print("注：显示的是各个link坐标系在base_link坐标系中的位姿（base_T_link）")
    print("=" * 80)
    
    for link_name, base_T_link in tf_dict.items():
        
        position = base_T_link[:3, 3]  # link原点在base_link坐标系中的位置
        rotation_matrix = base_T_link[:3, :3]  # 从base_link到link的旋转
        
        # 将旋转矩阵转换为RPY
        rot = R.from_matrix(rotation_matrix)
        rpy = rot.as_euler('xyz', degrees=True)
        
        print(f"🔗 {link_name}:")
        print(f"   位置 (相对于base_link): [{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}]")
        print(f"   方向 (RPY deg): [{rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f}]")
        
        # 计算从base_link原点的距离
        distance = np.linalg.norm(position)
        print(f"   距离base_link原点: {distance:.6f} m")
        print()


def hot_reload_monitor(urdf_path, callback, interval=2.0):
    """
    监控URDF文件变化并热重载
    
    Args:
        urdf_path: URDF文件路径
        callback: 回调函数，当文件变化时调用
        interval: 检查间隔（秒）
    """
    print(f"🔍 开始监控URDF文件: {urdf_path}")
    print(f"⏰ 检查间隔: {interval} 秒")
    print("🔄 按 Ctrl+C 停止监控")
    print()
    
    last_mtime = 0
    
    try:
        while True:
            current_mtime = Path(urdf_path).stat().st_mtime
            
            if current_mtime != last_mtime:
                print(f"\n📄 检测到URDF文件变化 ({time.strftime('%H:%M:%S')})")
                print("🔄 重新解析URDF文件...")
                
                try:
                    callback(urdf_path)
                    last_mtime = current_mtime
                except Exception as e:
                    print(f"❌ 解析失败: {e}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n👋 停止监控")


def process_urdf(urdf_path):
    """
    处理URDF文件的主要函数
    
    Args:
        urdf_path: URDF文件路径
    """
    print(f"📖 解析URDF文件: {urdf_path}")
    
    # 解析URDF
    links, joints = parse_urdf(urdf_path)
    
    print(f"✅ 解析完成:")
    print(f"   - 找到 {len(links)} 个link")
    print(f"   - 找到 {len(joints)} 个joint")
    
    # 打印link信息
    print("\n🔗 Link列表:")
    for link_name in links.keys():
        print(f"  - {link_name}")
    
    # 打印joint信息
    print("\n🔩 Joint列表:")
    for joint_name, joint_info in joints.items():
        print(f"  - {joint_name}: {joint_info['parent']} -> {joint_info['child']} ({joint_info['type']})")
        origin = joint_info['origin']
        print(f"    位置: [{origin['xyz'][0]:.6f}, {origin['xyz'][1]:.6f}, {origin['xyz'][2]:.6f}]")
        print(f"    方向: [{origin['rpy'][0]:.6f}, {origin['rpy'][1]:.6f}, {origin['rpy'][2]:.6f}] rad")
    
    # 计算tf树
    tf_dict = compute_tf_tree(joints)
    
    # 打印tf信息
    print_tf_info(tf_dict)
    
    # 可视化tf树
    visualize_tf_tree(tf_dict)
    
    return True


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='URDF TF可视化工具')
    parser.add_argument('--urdf', type=str, 
                       default="descriptions/genrobot/ego_v2.urdf",
                       help='URDF文件路径')
    parser.add_argument('--hot-reload', action='store_true',
                       help='启用热重载模式，监控URDF文件变化')
    parser.add_argument('--interval', type=float, default=2.0,
                       help='热重载检查间隔（秒）')
    
    args = parser.parse_args()
    
    urdf_path = args.urdf
    
    # 检查文件是否存在
    if not Path(urdf_path).exists():
        print(f"❌ 文件不存在: {urdf_path}")
        return
    
    if args.hot_reload:
        # 热重载模式
        hot_reload_monitor(urdf_path, process_urdf, args.interval)
    else:
        # 单次运行模式
        process_urdf(urdf_path)
        
        # 保持程序运行，让用户查看可视化结果
        print("\n🎯 可视化窗口已打开，按 Ctrl+C 退出...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 退出程序")


if __name__ == "__main__":
    main()