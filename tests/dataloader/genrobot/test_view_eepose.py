import mcap.reader
import json
import plotly.graph_objects as go
from collections import defaultdict
from pathlib import Path

# ================= 解码器模块 (参考你的代码) =================
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

def visualize_mcap_trajectory(mcap_file_path, topics_filter=None):
    if topics_filter is None:
        topics_filter = ["/robot0/vio/eef_pose", "/robot0/vio/relative_eef_pose"]

    trajectories = defaultdict(lambda: {"x": [], "y": [], "z": [], "timestamps": []})
    msg_count = 0
    parsed_count = 0

    print(f"📖 正在读取 MCAP 文件: {Path(mcap_file_path).name}")
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
                    trajectories[channel.topic]["x"].append(x)
                    trajectories[channel.topic]["y"].append(y)
                    trajectories[channel.topic]["z"].append(z)
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

    # ================= Plotly 可视化 =================
    print("🎨 正在生成 3D 轨迹图...")
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    topic_keys = list(trajectories.keys())

    for i, topic in enumerate(topic_keys):
        data = trajectories[topic]
        if not data["x"]:
            continue

        # 准备悬浮提示时间戳
        hover_ts = [f"{t/1e9:.3f} s" for t in data["timestamps"]]
        short_name = topic.split('/')[-1].replace('_', ' ').title()

        fig.add_trace(go.Scatter3d(
            x=data["x"], y=data["y"], z=data["z"],
            mode='lines',
            name=short_name,
            line=dict(width=3, color=colors[i % len(colors)]),
            text=hover_ts,
            hovertemplate="X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<br>Time: %{text}<extra></extra>"
        ))

        # 起点(●) / 终点(×) 标记
        fig.add_trace(go.Scatter3d(
            x=[data["x"][0], data["x"][-1]],
            y=[data["y"][0], data["y"][-1]],
            z=[data["z"][0], data["z"][-1]],
            mode='markers',
            marker=dict(size=8, color=colors[i % len(colors)], symbol=['circle', 'x']),
            name=f"{short_name} Endpoints",
            showlegend=False
        ))

    fig.update_layout(
        title="🤖 VIO 3D Trajectory (World & Relative EEF)",
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data',  # 🔑 保持真实物理比例，防止坐标轴拉伸变形
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, b=0, t=40),
        width=1100, height=750
    )

    fig.show()
    print("🚀 可视化完成。可在浏览器中旋转/缩放/悬浮查看数据。")

if __name__ == "__main__":
    # 替换为你的实际路径
    mcap_path = "/home/ryu-yang/Documents/Datasets/Domestic_Services/Living_Room/Organization/Organize_desktop/3a8f559dfb0847c8be710fa31c37758a.mcap"
    visualize_mcap_trajectory(mcap_path)
