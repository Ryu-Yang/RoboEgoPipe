"""
从 MCAP 文件中提取 camera2 和 camera3 的第一帧图像并保存为图片
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
import logging
from rich.logging import RichHandler

from roboegopipe.dataloader.genrobot import GenrobotdataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
log = logging.getLogger()


def save_first_frame(mcap_path: str, output_dir: str = "output_frames"):
    """
    读取 MCAP 文件中 camera2 和 camera3 的第一帧图像并保存为图片
    
    Args:
        mcap_path: MCAP 文件路径
        output_dir: 输出图片的目录
    """
    # 检查文件是否存在
    if not Path(mcap_path).exists():
        log.error(f"❌ 文件不存在: {mcap_path}")
        return
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log.info("🔧 初始化数据加载器...")
    dataLoader = GenrobotdataLoader(mcap_path)
    
    log.info("📖 读取数据并解码图像...")
    dataLoader.read_data(decode_images=True)
    
    # 获取所有图像数据
    images = dataLoader.decode_all_images()
    
    # 查找 camera2 和 camera3 的图像
    camera_topics = {}
    for topic, data in images.items():
        if 'camera2' in topic:
            camera_topics['camera2'] = (topic, data)
        elif 'camera3' in topic:
            camera_topics['camera3'] = (topic, data)
    
    if not camera_topics:
        log.error("❌ 未找到 camera2 或 camera3 的图像数据")
        log.error(f"可用的 topics: {list(images.keys())}")
        return
    
    # 保存每个相机的第一帧
    for cam_name, (topic, data) in camera_topics.items():
        img_list = data["images"]
        timestamps = data["timestamps"]
        
        if len(img_list) == 0:
            log.warning(f"⚠️ {cam_name} 没有图像数据")
            continue
        
        # 获取第一帧
        first_frame = img_list[0]
        first_timestamp = timestamps[0]
        
        # 将图像转换为正确的格式 (如果是浮点数则转换为 uint8)
        if first_frame.dtype == np.float32 or first_frame.dtype == np.float64:
            if first_frame.max() <= 1.0:
                first_frame = (first_frame * 255).astype(np.uint8)
            else:
                first_frame = first_frame.astype(np.uint8)
        
        # 如果是 BGR 格式直接保存，如果是其他格式需要转换
        if len(first_frame.shape) == 2:
            # 灰度图
            save_path = output_path / f"{cam_name}_first_frame.png"
            cv2.imwrite(str(save_path), first_frame)
        elif len(first_frame.shape) == 3:
            if first_frame.shape[2] == 3:
                # BGR 或 RGB 格式
                save_path = output_path / f"{cam_name}_first_frame.png"
                cv2.imwrite(str(save_path), first_frame)
            else:
                log.warning(f"⚠️ {cam_name} 图像通道数异常: {first_frame.shape}")
                continue
        else:
            log.warning(f"⚠️ {cam_name} 图像格式异常: {first_frame.shape}")
            continue
        
        log.info(f"✅ 已保存 {cam_name} 第一帧: {save_path}")
        log.info(f"   - 时间戳: {first_timestamp}")
        log.info(f"   - 图像尺寸: {first_frame.shape}")
    
    log.info(f"🎉 所有第一帧图像已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='提取 camera2 和 camera3 的第一帧图像')
    parser.add_argument('--mcap_path', type=str, 
                       default="data/3ff3c88df0e24b88a6a36232763bf21b.mcap",
                       help='MCAP文件路径')
    parser.add_argument('--output_dir', type=str,
                       default="output/read_output_frames",
                       help='输出图片的目录')
    
    args = parser.parse_args()
    
    save_first_frame(args.mcap_path, args.output_dir)


if __name__ == "__main__":
    main()