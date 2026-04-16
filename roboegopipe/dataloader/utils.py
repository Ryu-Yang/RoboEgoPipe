import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R


def parse_urdf(urdf_path):
    """
    解析URDF文件，提取所有link和joint的tf关系
    
    Args:
        urdf_path: URDF文件路径
        
    Returns:
        links: 字典，link名称 -> link信息
        joints: 字典，joint名称 -> joint信息（包含parent, child, origin变换）
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    links = {}
    joints = {}
    
    # 解析所有link
    for link_elem in root.findall('link'):
        link_name = link_elem.get('name')
        links[link_name] = {
            'name': link_name,
            'inertial': None,
            'visual': None,
            'collision': None
        }
        
        # 解析惯性参数
        inertial_elem = link_elem.find('inertial')
        if inertial_elem is not None:
            origin_elem = inertial_elem.find('origin')
            if origin_elem is not None:
                xyz = origin_elem.get('xyz', '0 0 0').split()
                rpy = origin_elem.get('rpy', '0 0 0').split()
                links[link_name]['inertial'] = {
                    'xyz': [float(x) for x in xyz],
                    'rpy': [float(r) for r in rpy]
                }
    
    # 解析所有joint
    for joint_elem in root.findall('joint'):
        joint_name = joint_elem.get('name')
        joint_type = joint_elem.get('type', 'fixed')
        
        parent_elem = joint_elem.find('parent')
        child_elem = joint_elem.find('child')
        
        if parent_elem is None or child_elem is None:
            continue
            
        parent_link = parent_elem.get('link')
        child_link = child_elem.get('link')
        
        # 解析origin变换
        origin_elem = joint_elem.find('origin')
        if origin_elem is not None:
            xyz = origin_elem.get('xyz', '0 0 0').split()
            rpy = origin_elem.get('rpy', '0 0 0').split()
            origin = {
                'xyz': [float(x) for x in xyz],
                'rpy': [float(r) for r in rpy]
            }
        else:
            origin = {'xyz': [0.0, 0.0, 0.0], 'rpy': [0.0, 0.0, 0.0]}
        
        joints[joint_name] = {
            'name': joint_name,
            'type': joint_type,
            'parent': parent_link,
            'child': child_link,
            'origin': origin
        }
    
    return links, joints

def rpy_to_matrix(rpy):
    """
    将RPY（roll, pitch, yaw）转换为4x4变换矩阵
    
    Args:
        rpy: [roll, pitch, yaw] in radians
        
    Returns:
        4x4变换矩阵
    """
    rot = R.from_euler('xyz', rpy).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    return T


def xyz_rpy_to_matrix(xyz, rpy):
    """
    将位置（xyz）和方向（rpy）转换为4x4变换矩阵
    
    Args:
        xyz: [x, y, z] 位置
        rpy: [roll, pitch, yaw] in radians
        
    Returns:
        4x4变换矩阵
    """
    rot = R.from_euler('xyz', rpy).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = xyz
    return T

def compute_tf_tree(joints, base_link: str = 'base_link'):
    """
    计算从各个link到base_link的变换矩阵（base_link_T_link）
     
    Args:
        joints: urdf joint字典
        base_link: base_link名称
        
    Returns:
        tf_dict: 字典，link名称 -> 从该link到base_link的4x4变换矩阵（base_link_T_link）
    """
    tf_dict = {base_link: np.eye(4)}
    
    # 使用BFS遍历tf树
    queue = [base_link]
    visited = set([base_link])
    
    while queue:
        current_link = queue.pop(0)
        
        # 查找所有以current_link为parent的joint
        for _joint_name, joint_info in joints.items():
            if joint_info['parent'] == current_link:
                child_link = joint_info['child']
                
                if child_link in visited:
                    continue
                    
                # joint的origin矩阵是parent_T_child（child相对于parent的位姿）
                parent_T_child = xyz_rpy_to_matrix(
                    joint_info['origin']['xyz'],
                    joint_info['origin']['rpy']
                )
                
                # 已知parent_T_base（从base到parent的变换）
                base_T_parent = tf_dict[current_link]
                
                # 计算child_T_base = child_T_parent @ parent_T_base
                base_T_child = base_T_parent @  parent_T_child
                
                tf_dict[child_link] = base_T_child
                queue.append(child_link)
                visited.add(child_link)
    
    return tf_dict
