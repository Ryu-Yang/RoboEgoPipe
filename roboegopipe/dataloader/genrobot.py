import json
import mcap.reader

from collections import defaultdict
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


_proto_classes = {}

def decode_message(schema, data):
    encoding = schema.encoding.lower() if schema else 'unknown'
    if encoding == 'json':
        return json.loads(data.decode('utf-8'))

    elif encoding == 'protobuf':
        return decode_protobuf_message(schema, data)
    else:
        raise ValueError(f"unsupport encoding: '{schema.encoding if schema else 'unknown'}'")

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
            raise ValueError(f"The message type cannot be found in the descriptor pool: {msg_type_name}")
        msg_class = message_factory.GetMessageClass(desc)
        _proto_classes[key] = msg_class

    msg = msg_class()
    msg.ParseFromString(data)
    return msg

def extract_pose_data(msg_obj, fallback_ts=None):
    """通用提取器：兼容 Dict (JSON)、Protobuf 对象、ROS2 对象"""
    try:
        if isinstance(msg_obj, dict):
            pose = msg_obj.get('pose', {})
            position = pose.get('position', {})
            orientation = pose.get('orientation', {})
            header = msg_obj.get('header', {})
            x, y, z = position.get('x'), position.get('y'), position.get('z')
            qx = orientation.get('x')
            qy = orientation.get('y')
            qz = orientation.get('z')
            qw = orientation.get('w')
            ts = header.get('timestamp')
        else:
            # 处理对象类型 (Protobuf / ROS2)
            position = msg_obj.pose.position
            orientation = msg_obj.pose.orientation if hasattr(msg_obj.pose, 'orientation') else None
            x, y, z = position.x, position.y, position.z
            if orientation:
                qx, qy, qz, qw = orientation.x, orientation.y, orientation.z, orientation.w
            else:
                qx = qy = qz = qw = None
            ts = msg_obj.header.timestamp if hasattr(msg_obj, 'header') else None

        # 统一处理时间戳 (可能是字符串、整数或 None)
        ts_ns = int(float(ts)) if ts is not None else fallback_ts

        if x is not None and y is not None and z is not None:
            return float(x), float(y), float(z), qx, qy, qz, qw, ts_ns
    except Exception:
        pass
    return None


class GenrobotdataLoader():
    def __init__(self, mcap_file_path: str, topics_filter: list[str] | None = None):
        self.mcap_file_path = mcap_file_path
        self.topics_filter = topics_filter

        if self.topics_filter is None:
            self.topics_filter = ["/robot0/vio/eef_pose", "/robot0/vio/relative_eef_pose"]

        # 存储轨迹数据（包含位置和方向）
        self.trajectories = defaultdict(lambda: {"positions": [], "orientations": [], "timestamps": []})
        self.msg_count = 0
        self.parsed_count = 0
    
    def read_data(self):

        print(f"📖 正在读取 MCAP 文件: {Path(self.mcap_file_path).name}")

        # 读取并解析数据
        with open(self.mcap_file_path, "rb") as f:
            reader = mcap.reader.make_reader(f)
            for schema, channel, message in reader.iter_messages():
                # 快速过滤 Topic
                if not any(t in channel.topic for t in self.topics_filter):
                    continue

                self.msg_count += 1
                try:
                    msg_obj = decode_message(schema, message.data)
                    pose_data = extract_pose_data(msg_obj, message.publish_time)
                    if pose_data:
                        x, y, z, qx, qy, qz, qw, ts_ns = pose_data
                        self.trajectories[channel.topic]["positions"].append([x, y, z])
                        if qx is not None and qy is not None and qz is not None and qw is not None:
                            self.trajectories[channel.topic]["orientations"].append([qx, qy, qz, qw])
                        else:
                            self.trajectories[channel.topic]["orientations"].append(None)
                        self.trajectories[channel.topic]["timestamps"].append(ts_ns)
                        self.parsed_count += 1
                except Exception as e:
                    if self.parsed_count == 0 and self.msg_count <= 5:
                        print(f"⚠️ 解析 {channel.topic} 失败 (前5条): {e}")
                    continue

        if self.msg_count == 0:
            print("❌ 未找到匹配的 Topic 数据。")
            return
        
        print(f"✅ 成功解析 {self.parsed_count}/{self.msg_count} 条位姿消息。")

        if not self.trajectories:
            print("❌ 未提取到有效坐标数据。")
            return

    def get_traj(self):
        return self.trajectories
