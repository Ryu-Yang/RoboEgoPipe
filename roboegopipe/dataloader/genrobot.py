import json
import mcap.reader
import cv2
import logging
import numpy as np
import re
import av
import io

from collections import defaultdict
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from scipy.spatial.transform import Rotation as R

from roboegopipe.dataloader.utils import (parse_urdf, rpy_to_matrix, xyz_rpy_to_matrix, 
    compute_tf_tree
)


log = logging.getLogger()


def get_T_body_imu():
    # URDF joint_imu origin values
    trans = np.array([0.0, 0.0, 0.0])
    rpy = np.array([3.14159265358979, 0.349741876975183, 0.0]) # Roll, Pitch, Yaw in radians
    
    # Create rotation matrix from RPY (scipy uses 'xyz' order by default for rpy)
    rot_mat = R.from_euler('xyz', rpy).as_matrix()
    
    # Construct 4x4 Transformation Matrix
    T_body_imu = np.eye(4)
    T_body_imu[:3, :3] = rot_mat
    T_body_imu[:3, 3] = trans
    
    return T_body_imu

def creat_matrix(pos, ori, input_type='quat', euler_seq='xyz'):
    """
    Create a 4x4 Homogeneous Transformation Matrix from position and orientation.

    Parameters:
    -----------
    pos : array-like, shape (3,)
        Position [x, y, z].
    ori : array-like, shape (3,) or (4,)
        Orientation. 
        - If input_type='euler': [roll, pitch, yaw] in radians.
        - If input_type='quat': [x, y, z, w] quaternion.
    input_type : str, optional
        'euler' or 'quat'. Default is 'euler'.
    euler_seq : str, optional
        Axis sequence for Euler angles (e.g., 'xyz', 'zyx'). Default is 'xyz'.

    Returns:
    --------
    matrix : np.ndarray, shape (4, 4)
        Homogeneous transformation matrix.
    """
    pos = np.asarray(pos)
    ori = np.asarray(ori)

    # 1. Handle Orientation -> Rotation Matrix (3x3)
    if input_type == 'euler':
        # scipy expects 'xyz' by default, but explicit is better
        rot = R.from_euler(euler_seq, ori)
    elif input_type == 'quat':
        # scipy expects [x, y, z, w]
        rot = R.from_quat(ori)
    else:
        raise ValueError("input_type must be 'euler' or 'quat'")

    rot_mat = rot.as_matrix()  # 3x3 Rotation Matrix

    # 2. Construct 4x4 Homogeneous Matrix
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = pos

    return T

def creat_pose(matrix):
    """
    将 4x4 变换矩阵转换为列表: [tx, ty, tz, qx, qy, qz, qw]
    """
    # matrix = np.asarray(matrix)
    
    # 1. 提取平移 (xyz)
    translation = matrix[:3, 3]
    
    # 2. 提取旋转并转为四元数 (xyzw)
    rotation_matrix = matrix[:3, :3]
    quat = R.from_matrix(rotation_matrix).as_quat()  # 返回 [qx, qy, qz, qw]
    
    # 3. 转换为普通 Python 列表并返回
    return translation.tolist() + quat.tolist()


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


def extract_camera_info(msg_obj, fallback_ts=None):
    """提取相机信息：兼容 Dict (JSON)、Protobuf 对象"""
    try:
        if isinstance(msg_obj, dict):
            # 从JSON格式提取相机信息
            camera_info = {
                'D': msg_obj.get('D', []),
                'K': msg_obj.get('K', []),
                'R': msg_obj.get('R', []),
                'P': msg_obj.get('P', []),
                'T_b_c': msg_obj.get('T_b_c', []),
                'width': msg_obj.get('width', 0),
                'height': msg_obj.get('height', 0),
                'distortion_model': msg_obj.get('distortion_model', ''),
                'frame_id': msg_obj.get('frame_id', ''),
                'header': msg_obj.get('header', {})
            }
            ts = msg_obj.get('header', {}).get('timestamp')
        else:
            # 处理对象类型 (Protobuf / ROS2)
            camera_info = {
                'D': list(msg_obj.D) if hasattr(msg_obj, 'D') else [],
                'K': list(msg_obj.K) if hasattr(msg_obj, 'K') else [],
                'R': list(msg_obj.R) if hasattr(msg_obj, 'R') else [],
                'P': list(msg_obj.P) if hasattr(msg_obj, 'P') else [],
                'T_b_c': list(msg_obj.T_b_c) if hasattr(msg_obj, 'T_b_c') else [],
                'width': msg_obj.width if hasattr(msg_obj, 'width') else 0,
                'height': msg_obj.height if hasattr(msg_obj, 'height') else 0,
                'distortion_model': msg_obj.distortion_model if hasattr(msg_obj, 'distortion_model') else '',
                'frame_id': msg_obj.frame_id if hasattr(msg_obj, 'frame_id') else '',
                'header': {
                    'timestamp': msg_obj.header.timestamp if hasattr(msg_obj, 'header') else None
                }
            }
            ts = msg_obj.header.timestamp if hasattr(msg_obj, 'header') else None

        # 统一处理时间戳
        ts_ns = int(float(ts)) if ts is not None else fallback_ts
        camera_info['timestamp'] = ts_ns

        # #处理T_b_c，这里得到的T_b_c实际上是T_imu_c,T_b_c = T_body_imu * T_imu_camera

        # T_body_imu = get_T_body_imu()

        # T_imu_camera = creat_matrix(
        #     [camera_info['T_b_c'][0], camera_info['T_b_c'][1], camera_info['T_b_c'][2]],
        #     [camera_info['T_b_c'][3], camera_info['T_b_c'][4], camera_info['T_b_c'][5], camera_info['T_b_c'][6]]
        # )
        # # T_b_c = T_body_imu @ T_imu_camera
        # T_b_c = T_imu_camera

        # #我这里需要把变换矩阵按照，xyz，xyzw存放
        # camera_info['T_b_c'] = creat_pose(T_b_c)
        # print(f"camera_info['T_b_c']: {camera_info['T_b_c']}")

        # 验证必要的相机参数
        if camera_info['K'] and len(camera_info['K']) >= 9:
            return camera_info
    except Exception:
        pass
    return None


def extract_compressed_image(msg_obj, fallback_ts=None):
    """提取压缩图像数据：兼容 Dict (JSON)、Protobuf 对象"""
    try:
        if isinstance(msg_obj, dict):
            # 从JSON格式提取图像数据
            data = msg_obj.get('data', [])
            format_str = msg_obj.get('format', '')
            frame_id = msg_obj.get('frame_id', '')
            header = msg_obj.get('header', {})
            ts = header.get('timestamp')
        else:
            # 处理对象类型 (Protobuf / ROS2)
            data = list(msg_obj.data) if hasattr(msg_obj, 'data') else []
            format_str = msg_obj.format if hasattr(msg_obj, 'format') else ''
            frame_id = msg_obj.frame_id if hasattr(msg_obj, 'frame_id') else ''
            ts = msg_obj.header.timestamp if hasattr(msg_obj, 'header') else None

        # 统一处理时间戳
        ts_ns = int(float(ts)) if ts is not None else fallback_ts
        
        # 将数据转换为字节数组
        if data:
            image_data = bytes(data)
            return {
                'data': image_data,
                'format': format_str,
                'frame_id': frame_id,
                'timestamp': ts_ns
            }
    except Exception:
        pass
    return None


# def decode_compressed_image(image_data, format_str='h264'):
#     """解码压缩图像数据"""
#     try:
#         if not image_data:
#             return None
            
#         # 根据格式选择解码方式
#         format_lower = format_str.lower()
        
#         if format_lower in ['h264', 'h265', 'hevc']:
#             # 使用OpenCV解码视频帧
#             # 将字节数据转换为numpy数组
#             np_arr = np.frombuffer(image_data, dtype=np.uint8)
            
#             # 创建解码器
#             if format_lower == 'h264':
#                 fourcc = cv2.VideoWriter_fourcc(*'H264')
#             elif format_lower == 'h265' or format_lower == 'hevc':
#                 fourcc = cv2.VideoWriter_fourcc(*'HEVC')
#             else:
#                 fourcc = cv2.VideoWriter_fourcc(*'H264')
            
#             # 尝试解码单帧
#             # 注意：对于H.264/H.265，可能需要更复杂的处理
#             # 这里使用简单的方法尝试解码
#             try:
#                 # 方法1：尝试使用imdecode
#                 img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#                 if img is not None:
#                     return img
#             except:
#                 pass
            
#             # 方法2：尝试创建临时视频解码器
#             try:
#                 # 创建临时文件来解码
#                 import tempfile
#                 with tempfile.NamedTemporaryFile(suffix='.h264', delete=False) as tmp:
#                     tmp.write(image_data)
#                     tmp.flush()
                    
#                     cap = cv2.VideoCapture(tmp.name)
#                     if cap.isOpened():
#                         ret, frame = cap.read()
#                         cap.release()
#                         if ret:
#                             return frame
#             except:
#                 pass
            
#             # 如果以上方法都失败，返回原始数据
#             return None
            
#         elif format_lower in ['jpeg', 'jpg']:
#             # JPEG解码
#             np_arr = np.frombuffer(image_data, dtype=np.uint8)
#             img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#             return img
            
#         elif format_lower in ['png']:
#             # PNG解码
#             np_arr = np.frombuffer(image_data, dtype=np.uint8)
#             img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#             return img
            
#         else:
#             # 未知格式，尝试通用解码
#             np_arr = np.frombuffer(image_data, dtype=np.uint8)
#             img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#             return img
            
#     except Exception as e:
#         print(f"⚠️ 解码图像失败: {e}")
#         return None

def decode_compressed_image(image_data: bytes, format_str: str = 'h264') -> np.ndarray | None:
    if not image_data:
        return None

    format_lower = format_str.lower()

    try:
        if format_lower in ['jpeg', 'jpg', 'png']:
            np_arr = np.frombuffer(image_data, dtype=np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        elif format_lower in ['h264', 'h265', 'hevc']:
            # 使用 PyAV 解码
            container = av.open(io.BytesIO(image_data), format=format_lower if format_lower != 'hevc' else 'hevc')
            stream = container.streams.video[0]
            
            for frame in container.decode(stream):
                # 转换为 BGR numpy 数组
                img = frame.to_ndarray(format='bgr24')
                return img
            
            return None

        else:
            np_arr = np.frombuffer(image_data, dtype=np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    except Exception as e:
        print(f"⚠️ 解码图像失败: {e}")
        return None


class BatchVideoDecoder:
    """批量视频解码器，专门处理H.264/H.265的批量解码"""
    def __init__(self, format_str='h264'):
        self.format = format_str.lower()
        
    def decode_frames(self, frames_data: list[bytes], target_size: tuple = (1600, 1300)) -> list[np.ndarray]:
        """批量解码视频帧
        
        Args:
            frames_data: 帧数据列表，每个元素是bytes
            
        Returns:
            解码后的图像列表
        """
        decoded_frames = []
        
        if not frames_data:
            return decoded_frames
            
        if self.format in ['h264', 'h265', 'hevc']:
            # 将所有帧数据合并为一个连续的字节流
            # 注意：假设frames_data中的每个元素都是完整的NAL单元
            combined_data = b''.join(frames_data)
            
            try:
                # 使用PyAV进行连续解码
                container_format = self.format if self.format != 'hevc' else 'hevc'
                container = av.open(io.BytesIO(combined_data), format=container_format)
                
                if not container.streams.video:
                    print(f"⚠️ 未找到视频流，格式: {self.format}")
                    return decoded_frames
                    
                stream = container.streams.video[0]
                
                for frame in container.decode(stream):
                    try:
                        img = frame.to_ndarray(format='bgr24')

                        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                        
                        decoded_frames.append(resized_img)
                    except Exception as e:
                        print(f"⚠️ 转换帧数据失败: {e}")
                        continue
                        
            except Exception as e:
                print(f"⚠️ 批量解码失败 ({self.format}): {e}")
                # 尝试逐帧解码作为后备方案
                return self._decode_frames_fallback(frames_data)
                
        else:
            # 对于非视频格式，使用逐帧解码
            return self._decode_frames_fallback(frames_data)
            
        return decoded_frames
    
    def _decode_frames_fallback(self, frames_data: list[bytes]) -> list[np.ndarray]:
        """后备方案：逐帧解码"""
        decoded_frames = []
        
        for i, frame_data in enumerate(frames_data):
            try:
                img = decode_compressed_image(frame_data, self.format)
                if img is not None:
                    decoded_frames.append(img)
            except Exception as e:
                print(f"⚠️ 解码第 {i} 帧失败: {e}")
                continue
                
        return decoded_frames


class GenrobotdataLoader():
    def __init__(self, mcap_file_path: str, topics_filter: list[str] | None = None):
        self.mcap_file_path = mcap_file_path
        self.topics_filter = topics_filter

        if self.topics_filter is None:
            self.topics_filter = ["/robot0/vio/eef_pose", "/robot0/vio/relative_eef_pose"]

        # 存储轨迹数据（包含位置和方向）
        self.trajectories = defaultdict(lambda: {"positions": [], "orientations": [], "timestamps": []})
        # 存储相机信息
        self.camera_info = defaultdict(lambda: {"info": [], "timestamps": []})
        # 存储压缩图像数据
        self.compressed_images = defaultdict(lambda: {"data": [], "format": [], "frame_id": [], "timestamps": []})
        # 存储解码后的图像
        self.decoded_images = defaultdict(lambda: {"images": [], "timestamps": []})
        
        self.traj_count = 0
        self.camera_count = 0
        self.image_count = 0
    
    def read_data(self, decode_images=False):
        """
        读取MCAP文件数据
        
        Args:
            decode_images: 是否标记需要解码图像数据（实际解码在后续批量进行）
        """
        log.info(f"📖 正在读取 MCAP 文件: {Path(self.mcap_file_path).name}")
        
        # 存储是否需要解码的标志
        self._need_decode_images = decode_images

        # 读取并解析数据
        with open(self.mcap_file_path, "rb") as f:
            reader = mcap.reader.make_reader(f)
            for schema, channel, message in reader.iter_messages():
                # 检查是否为相机信息topic
                is_camera_info_topic = '/sensor/camera' in channel.topic and '/camera_info' in channel.topic
                # 检查是否为压缩图像topic
                is_compressed_image_topic = '/sensor/camera' in channel.topic and '/compressed' in channel.topic
                is_pose_topic = any(t in channel.topic for t in self.topics_filter)
                
                if not (is_camera_info_topic or is_compressed_image_topic or is_pose_topic):
                    continue

                try:
                    msg_obj = decode_message(schema, message.data)
                    
                    if is_camera_info_topic:
                        # 处理相机信息
                        camera_data = extract_camera_info(msg_obj, message.publish_time)
                        if camera_data:
                            self.camera_info[channel.topic]["info"].append(camera_data)
                            self.camera_info[channel.topic]["timestamps"].append(camera_data['timestamp'])
                            self.camera_count += 1
                    
                    elif is_compressed_image_topic:
                        # 处理压缩图像数据
                        image_data = extract_compressed_image(msg_obj, message.publish_time)
                        if image_data:
                            self.compressed_images[channel.topic]["data"].append(image_data['data'])
                            self.compressed_images[channel.topic]["format"].append(image_data['format'])
                            self.compressed_images[channel.topic]["frame_id"].append(image_data['frame_id'])
                            self.compressed_images[channel.topic]["timestamps"].append(image_data['timestamp'])
                            self.image_count += 1
                    
                    elif is_pose_topic:
                        # 处理位姿数据
                        pose_data = extract_pose_data(msg_obj, message.publish_time)
                        if pose_data:
                            x, y, z, qx, qy, qz, qw, ts_ns = pose_data
                            self.trajectories[channel.topic]["positions"].append([x, y, z])
                            if qx is not None and qy is not None and qz is not None and qw is not None:
                                self.trajectories[channel.topic]["orientations"].append([qx, qy, qz, qw])
                            else:
                                self.trajectories[channel.topic]["orientations"].append(None)
                            self.trajectories[channel.topic]["timestamps"].append(ts_ns)
                            self.traj_count += 1

                    if self.traj_count >=500:
                        break
                            
                except Exception as e:
                    log.warning(f"⚠️ 解析 {channel.topic} 失败: {e}")
                    continue
        if self.traj_count > 0:
            log.info(f"✅ 成功解析 {self.traj_count} 条位姿消息。")

        if self.camera_count > 0:
            log.info(f"📷 成功解析 {self.camera_count} 条相机信息。")
            for topic, data in self.camera_info.items():
                if data["info"]:
                    short_name = topic.split('/')[-2] + '/' +topic.split('/')[-1]
                    log.info(f"  - {short_name}: {len(data['info'])} 条信息")
        
        if self.image_count > 0:
            log.info(f"📸 成功解析 {self.image_count} 条压缩图像消息。")
            for topic, data in self.compressed_images.items():
                if data["data"]:
                    short_name = topic.split('/')[-2] + '/' + topic.split('/')[-1]
                    log.info(f"  - {short_name}: {len(data['data'])} 张图像, 格式: {data['format'][0] if data['format'] else 'unknown'}")
            
            if decode_images:
                decoded_count = sum(len(data["images"]) for data in self.decoded_images.values())
                log.info(f"  - 已解码 {decoded_count} 张图像")

        if not self.trajectories and not self.camera_info and not self.compressed_images:
            log.error("❌ 未提取到有效数据。")
            return

    def get_traj(self):
        return self.trajectories
    
    def get_camera_info(self, from_urdf: bool, urdf_path: str | None = None):

        if from_urdf and urdf_path:
            self.camera_link = defaultdict(lambda: {"info": [], "timestamps": []})
            
            
            _links, joints = parse_urdf(urdf_path)

            tf_dict = compute_tf_tree(joints)
            

            for link_name, base_T_link in tf_dict.items():
                if 'camera' in link_name:

                    match = re.search(r'\d+', link_name)
        
                    if not match:
                        continue 
                        
                    num_str = match.group(0)
                    
                    match_name = f"camera{num_str}"

                    for key, _data in self.camera_info.items():
                        if match_name in key:
                            match_name = key
                            break

                    camera_info = self.camera_info[match_name]["info"][0]
                    camera_info['T_b_c'] = creat_pose(base_T_link)
                
                    self.camera_link[link_name]["info"].append(camera_info)

            return self.camera_link
        else:
            return self.camera_info


    def get_compressed_images(self):
        return self.compressed_images
    
    def get_decoded_images(self):
        return self.decoded_images
    
    def decode_all_images(self):
        """解码所有压缩图像（使用批量解码）"""
        return self.decode_images_batch()
    
    def decode_images_batch(self):
        """批量解码所有压缩图像（优化版）"""
        print("🔄 开始批量解码所有压缩图像...")
        total_decoded = 0
        
        for topic, data in self.compressed_images.items():
            images = data["data"]
            formats = data["format"]
            timestamps = data["timestamps"]
            
            if not images:
                continue
            
            # 获取话题的短名称用于日志
            short_name = topic.split('/')[-2] if len(topic.split('/')) >= 2 else topic
            
            # 检查格式是否一致
            format_set = set(formats)
            if len(format_set) > 1:
                print(f"⚠️ 话题 {short_name} 包含多种格式: {format_set}")
                # 按格式分组处理
                from collections import defaultdict
                format_groups = defaultdict(list)
                for i, fmt in enumerate(formats):
                    format_groups[fmt].append((i, images[i], timestamps[i]))
                
                decoded_list = []
                timestamp_list = []
                
                for fmt, group in format_groups.items():
                    indices = [item[0] for item in group]
                    frame_data = [item[1] for item in group]
                    group_timestamps = [item[2] for item in group]
                    
                    decoder = BatchVideoDecoder(fmt)
                    decoded_frames = decoder.decode_frames(frame_data)
                    
                    # 按原始顺序重新组合
                    for idx, frame, ts in zip(indices, decoded_frames, group_timestamps):
                        decoded_list.append((idx, frame))
                        timestamp_list.append((idx, ts))
                        total_decoded += 1
                
                # 排序
                decoded_list.sort(key=lambda x: x[0])
                timestamp_list.sort(key=lambda x: x[0])
                
                self.decoded_images[topic]["images"] = [frame for _, frame in decoded_list]
                self.decoded_images[topic]["timestamps"] = [ts for _, ts in timestamp_list]
            else:
                # 单一格式，直接批量解码
                fmt = formats[0]
                decoder = BatchVideoDecoder(fmt)
                decoded_frames = decoder.decode_frames(images)
                
                self.decoded_images[topic]["images"] = decoded_frames
                # 确保时间戳数量与解码图像数量一致
                self.decoded_images[topic]["timestamps"] = timestamps[:len(decoded_frames)]
                total_decoded += len(decoded_frames)
            
            decoded_count = len(self.decoded_images[topic]["images"])
            print(f"  ✅ {short_name}: 解码了 {decoded_count}/{len(images)} 张图像")
        
        print(f"✅ 成功批量解码 {total_decoded} 张图像")
        return self.decoded_images
    
    def get_image_by_timestamp(self, topic, timestamp_ns, tolerance_ns=1000000):
        """
        根据时间戳获取图像
        
        Args:
            topic: 图像topic
            timestamp_ns: 目标时间戳（纳秒）
            tolerance_ns: 时间戳容差（纳秒）
        
        Returns:
            解码后的图像或None
        """
        if topic not in self.decoded_images:
            # 如果没有解码的图像，尝试解码
            if topic in self.compressed_images:
                self.decode_all_images()
            else:
                return None
        
        images = self.decoded_images[topic]["images"]
        timestamps = self.decoded_images[topic]["timestamps"]
        
        if not images:
            return None
        
        # 寻找最接近时间戳的图像
        min_diff = float('inf')
        best_idx = -1
        
        for i, ts in enumerate(timestamps):
            diff = abs(ts - timestamp_ns)
            if diff < min_diff:
                min_diff = diff
                best_idx = i
        
        if best_idx >= 0 and min_diff <= tolerance_ns:
            return images[best_idx]
        
        return None
