"""
使用Rerun可视化MCAP文件中的机器人末端执行器位姿轨迹
替换原有的Plotly可视化方案
"""

import mcap.reader
import json
import rerun as rr
import numpy as np
from collections import defaultdict
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional

# ================= 解码器模块 (参考原有代码) =================
try:
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    protobuf_available = True
except ImportError:
    protobuf_available = False

try:
    from mcap_ros2.decoder import decode_ros2_message
    ros2_available = True
except ImportError:
    ros2_available = False

_proto_classes = {}

def decode_message(schema, data):
    encoding = schema.encoding.lower() if schema else 'unknown'
    if encoding == 'json':
        return json.loads(data.decode('utf-8'))
    elif encoding == 'ros2msg':
        if not ros2_available:
            raise ImportError("缺少 mcap_ros2 模块，无法解码 ROS2 消息")
        return decode_ros2_message(schema, data)
    elif encoding == 'protobuf':
        if not protobuf_available:
            raise ImportError("缺少 protobuf 模块，无法解码 protobuf 消息")
        return decode_protobuf_message(schema, data)
    else:
        raise ValueError(f"不支持的编码: '{schema.encoding if schema else 'unknown'}'")

def decode_protobuf_message(schema, data):
    msg_type_name = schema.name
    pool = descriptor_pool.Default()
    key = (schema.id, msg_type_name)

    if key in _proto_classes:
        msg_class = _proto_classes[key]
    else:
        fds = descriptor_pb2.FileDescriptorSet()
        fds.ParseFromString(schema.data)
        for f in fds.file:
            pool.Add(f)
        try:
            desc = pool.FindMessageTypeByName(msg_type_name)
        except KeyError:
            raise ValueError(f"无法在描述符池中找到消息类型: {msg_type_name}")
        msg_class = message_factory.GetMessageClass(desc)
        _proto_classes[key] = msg_class

    msg = msg_class()
    msg.ParseFromString(data)
    return msg
# ==========================================================

def extract_pose_data(msg_obj, fallback_ts=None):
    """通用提取器：兼容 Dict (JSON)、Protobuf 对象、ROS2 对象"""
    try:
        if isinstance(msg_obj, dict):
            pose = msg_obj.get('pose', {})
            position = pose.get('position', {})
            header = msg_obj.get('header', {})
            x, y, z = position.get('x'), position.get('y'), position.get('z')
            ts = header.get('timestamp')
        else:
            # 处理对象类型 (Protobuf / ROS2)
            position = msg_obj.pose.position
            x, y, z = position.x, position.y, position.z
            ts = msg_obj.header.timestamp if hasattr(msg_obj, 'header') else None

        # 统一处理时间戳 (可能是字符串、整数或 None)
        ts_ns = int(float(ts)) if ts is not None else fallback_ts

        if x is not None and y is not None and z is not None:
            return float(x), float(y), float(z), ts_ns
    except Exception:
        pass
    return None

def visualize_mcap_trajectory_with_rerun(mcap_file_path: str, topics_filter: Optional[List[str]] = None):
    """
    使用Rerun可视化MCAP文件中的轨迹
    
    Args:
        mcap_file_path: MCAP文件路径
        topics_filter: 要可视化的topic列表，默认为None时使用默认topic
    """
    if topics_filter is None:
        topics_filter = ["/robot0/vio/eef_pose", "/robot0/vio/relative_eef_pose"]

    # 存储轨迹数据
    trajectories = defaultdict(lambda: {"positions": [], "timestamps": []})
    msg_count = 0
    parsed_count = 0

    print(f"📖 正在读取 MCAP 文件: {Path(mcap_file_path).name}")
    
    # 读取并解析数据
    with open(mcap_file_path, "rb") as f:
        reader = mcap.reader.make_reader(f)
        for schema, channel, message in reader.iter_messages():
            # 快速过滤 Topic
            if not any(t in channel.topic for t in topics_filter):
                continue

            msg_count += 1
            try:
                msg_obj = decode_message(schema, message.data)
                pose_data = extract_pose_data(msg_obj, message.publish_time)
                if pose_data:
                    x, y, z, ts_ns = pose_data
                    trajectories[channel.topic]["positions"].append([x, y, z])
                    trajectories[channel.topic]["timestamps"].append(ts_ns)
                    parsed_count += 1
            except Exception as e:
                if parsed_count == 0 and msg_count <= 5:
                    print(f"⚠️ 解析 {channel.topic} 失败 (前5条): {e}")
                continue

    if msg_count == 0:
        print("❌ 未找到匹配的 Topic 数据。")
        return
    
    print(f"✅ 成功解析 {parsed_count}/{msg_count} 条位姿消息。")

    if not trajectories:
        print("❌ 未提取到有效坐标数据。")
        return

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
        
        # 简化topic名称用于显示
        short_name = topic.split('/')[-1].replace('_', ' ').title()
        
        # 创建轨迹实体路径
        entity_path = f"world/trajectories/{short_name}"
        
        # 记录轨迹线
        rr.log(
            f"{entity_path}/line",
            rr.LineStrips3D([positions], colors=color),
            static=True
        )
        
        # 记录轨迹点（带时间戳）
        if len(timestamps) > 0:
            # 转换为秒
            timestamps_seconds = timestamps / 1e9
            
            # 按时间顺序记录每个点
            for idx, (pos, ts_ns) in enumerate(zip(positions, timestamps)):
                rr.set_time("timestamp", timestamp=ts_ns)
                rr.log(
                    f"{entity_path}/points",
                    rr.Points3D([pos], radii=0.01, colors=color)
                )
                
                # 每100个点记录一次，避免过多数据点
                if idx % 100 == 0:
                    rr.log(
                        f"{entity_path}/points_batch",
                        rr.Points3D(positions[:idx+1], radii=0.005, colors=color)
                    )
        
        # 记录起点和终点
        rr.set_time("timestamp", timestamp=0)
        rr.log(
            f"{entity_path}/start",
            rr.Points3D([positions[0]], radii=0.02, colors=[0, 255, 0])  # 绿色起点
        )
        
        if len(positions) > 1:
            rr.log(
                f"{entity_path}/end",
                rr.Points3D([positions[-1]], radii=0.02, colors=[255, 0, 0])  # 红色终点
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

def main():
    """主函数"""
    # 替换为你的实际路径
    mcap_path = "/home/ryu-yang/Documents/Datasets/Domestic_Services/Living_Room/Organization/Organize_desktop/3a8f559dfb0847c8be710fa31c37758a.mcap"
    
    # 检查文件是否存在
    if not Path(mcap_path).exists():
        print(f"❌ 文件不存在: {mcap_path}")
        print("请修改代码中的 mcap_path 变量为有效的MCAP文件路径")
        return
    
    # 检查Rerun是否可用
    try:
        import rerun as rr
        print("✅ Rerun库可用")
    except ImportError:
        print("❌ Rerun库未安装")
        print("请安装: pip install rerun-sdk")
        return
    
    visualize_mcap_trajectory_with_rerun(mcap_path)

if __name__ == "__main__":
    main()