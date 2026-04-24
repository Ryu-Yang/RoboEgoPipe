import numpy as np
import cv2

def double_sphere_project(points_3d, fx, fy, cx, cy, xi, alpha):
    """
    双球模型投影（批量）
    points_3d: (N, 3) 相机坐标系下的3D点，z > 0
    返回: (N, 2) 像素坐标
    """
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    
    d1 = np.sqrt(x**2 + y**2 + z**2)
    k = xi * d1 + z
    d2 = np.sqrt(x**2 + y**2 + k**2)
    denom = alpha * d2 + (1 - alpha) * k
    mx = x / denom
    my = y / denom
    
    u = fx * mx + cx
    v = fy * my + cy
    return np.column_stack((u, v))


def generate_pinhole_to_double_sphere_map(fx_ds, fy_ds, cx_ds, cy_ds, xi, alpha,
                                           fx_pin, fy_pin, cx_pin, cy_pin,
                                           output_width, output_height):
    """
    生成从虚拟针孔图像到原始双球图像的重映射表
    
    返回: map_x, map_y 可直接用于 cv2.remap
    """
    # 生成输出图像的网格坐标
    u_grid, v_grid = np.meshgrid(np.arange(output_width), np.arange(output_height))
    u_flat = u_grid.flatten().astype(np.float32)
    v_flat = v_grid.flatten().astype(np.float32)
    
    # 虚拟针孔相机：像素 → 归一化平面坐标 → 3D方向向量
    mx = (u_flat - cx_pin) / fx_pin
    my = (v_flat - cy_pin) / fy_pin
    mz = np.ones_like(mx)
    
    # 归一化方向向量
    norm = np.sqrt(mx*mx + my*my + mz*mz)
    rays = np.column_stack((mx/norm, my/norm, mz/norm))
    
    # 将该射线投影到双球模型图像平面
    pixels_ds = double_sphere_project(rays, fx_ds, fy_ds, cx_ds, cy_ds, xi, alpha)
    
    # 重塑为映射表格式
    map_x = pixels_ds[:, 0].reshape(output_height, output_width)
    map_y = pixels_ds[:, 1].reshape(output_height, output_width)
    
    return map_x, map_y


def remap_to_pinhole(image_ds, 
                      fx_ds, fy_ds, cx_ds, cy_ds, xi, alpha,
                      fx_pin, fy_pin, cx_pin, cy_pin,
                      output_size):
    """
    将双球模型图像重映射到针孔模型图像
    image_ds: 原始双球模型图像
    output_size: (width, height) 输出针孔图像尺寸
    """
    output_width, output_height = output_size
    
    # 生成映射表
    map_x, map_y = generate_pinhole_to_double_sphere_map(
        fx_ds, fy_ds, cx_ds, cy_ds, xi, alpha,
        fx_pin, fy_pin, cx_pin, cy_pin,
        output_width, output_height
    )
    
    # 应用重映射
    output = cv2.remap(image_ds, map_x, map_y, cv2.INTER_LINEAR)
    return output


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 你的双球模型参数
    fx_ds, fy_ds = 500.0, 500.0
    cx_ds, cy_ds = 640.0, 480.0
    xi = -0.0055
    alpha = 0.567
    
    # 虚拟针孔相机的内参（可以根据需要设置）
    # 推荐：保持中心不变，焦距适当调整以保留尽可能多的视场角
    fx_pin, fy_pin = 400.0, 400.0   # 可以适当减小来保留更大视场角
    cx_pin, cy_pin = 640.0, 480.0   # 与双球模型一致
    
    # 输出尺寸
    output_size = (1280, 960)
    
    # 读取原始图像（双球模型拍摄的）
    image_ds = cv2.imread("fisheye_image.jpg")
    
    # 重映射到针孔
    image_pinhole = remap_to_pinhole(
        image_ds, 
        fx_ds, fy_ds, cx_ds, cy_ds, xi, alpha,
        fx_pin, fy_pin, cx_pin, cy_pin,
        output_size
    )
    
    # 保存结果
    cv2.imwrite("pinhole_output.jpg", image_pinhole)