import numpy as np
import cv2

from roboegopipe.cammodel.double_sphere import DoubleSphereCameraModel

# ========== 使用示例 ==========
if __name__ == "__main__":
    # # 相机左的内参
    # # 双球模型参数（请替换为你的实际标定值）
    # fx_ds, fy_ds = 507.87387315510574, 509.4144511814894
    # cx_ds, cy_ds = 796.7150518282281, 655.6687369325308
    # xi = -0.005512095399300616
    # alpha = 0.567063669804753
    
    # # 虚拟针孔相机内参
    # fx_pin, fy_pin = 507.87387315510574, 509.4144511814894
    # cx_pin, cy_pin = 796.7150518282281, 655.6687369325308

    # 相机右的内参 
    # 双球模型参数（请替换为你的实际标定值）
    fx_ds, fy_ds = 511.053073270337, 512.7060865113444
    cx_ds, cy_ds = 804.5116352623274, 617.9051244593389
    xi = -0.005484440261185868
    alpha = 0.5678308272102589
    
    # 虚拟针孔相机内参
    fx_pin, fy_pin = 511.053073270337, 512.7060865113444
    cx_pin, cy_pin = 804.5116352623274, 617.9051244593389
    
    output_size = (1600, 1300)

    cam_model = DoubleSphereCameraModel(fx_ds, fy_ds, cx_ds, cy_ds, xi, alpha)
    
    image_ds = cv2.imread("output/read_output_frames/camera3_first_frame.png")
    if image_ds is None:
        raise FileNotFoundError("请提供有效的输入图像路径")
        
    image_pinhole = cam_model.remap_to_pinhole(
        image_ds,
        fx_pin, fy_pin, cx_pin, cy_pin, output_size
    )
    
    cv2.imwrite("output/undistorted_images/r_pinhole_output.jpg", image_pinhole)
    print("✅ 重映射完成。新内参 K = [[fx_pin, 0, cx_pin], [0, fy_pin, cy_pin], [0, 0, 1]]")
    print("✅ 新畸变参数 D = [0, 0, 0, 0]")