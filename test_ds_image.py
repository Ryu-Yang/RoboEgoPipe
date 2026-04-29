import numpy as np
import cv2

from roboegopipe.cammodel.double_sphere import DoubleSphereCameraModel

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 相机左的内参
    # 双球模型参数（请替换为你的实际标定值）
    fx_ds_l, fy_ds_l = 507.87387315510574, 509.4144511814894
    cx_ds_l, cy_ds_l = 796.7150518282281, 655.6687369325308
    xi_l = -0.005512095399300616
    alpha_l = 0.567063669804753

    # 相机右的内参 
    # 双球模型参数（请替换为你的实际标定值）
    fx_ds_r, fy_ds_r = 511.053073270337, 512.7060865113444
    cx_ds_r, cy_ds_r = 804.5116352623274, 617.9051244593389
    xi_r = -0.005484440261185868
    alpha_r = 0.5678308272102589
    
    # 虚拟针孔相机内参
    fx_pin, fy_pin = 360, 360
    cx_pin, cy_pin = 480, 360
    
    output_size = (960, 720)

    cam_model_l = DoubleSphereCameraModel(fx_ds_l, fy_ds_l, cx_ds_l, cy_ds_l, xi_l, alpha_l)
    cam_model_r = DoubleSphereCameraModel(fx_ds_r, fy_ds_r, cx_ds_r, cy_ds_r, xi_r, alpha_r)
    
    image_ds_l = cv2.imread("output/read_output_frames/camera2_first_frame.png")
    image_ds_r = cv2.imread("output/read_output_frames/camera3_first_frame.png")
    if image_ds_l is None or image_ds_r is None:
        raise FileNotFoundError("请提供有效的输入图像路径")
        
    image_pinhole_l = cam_model_l.remap_to_pinhole(
        image_ds_l,
        fx_pin, fy_pin, cx_pin, cy_pin, output_size
    )
    image_pinhole_r = cam_model_r.remap_to_pinhole(
        image_ds_r,
        fx_pin, fy_pin, cx_pin, cy_pin, output_size
    )
    
    cv2.imwrite("output/new_undistorted_images/l_pinhole_output.jpg", image_pinhole_l)
    cv2.imwrite("output/new_undistorted_images/r_pinhole_output.jpg", image_pinhole_r)

    print(f"✅ 重映射完成。新内参 K = [[{fx_pin}, 0, {cx_pin}], [0, {fy_pin}, {cy_pin}], [0, 0, 1]]")
    print(f"✅ 新畸变参数 D = [0, 0, 0, 0]")