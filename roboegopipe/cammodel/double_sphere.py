import numpy as np
import cv2

class DoubleSphereCameraModel():
    def __init__(self, fx, fy, cx, cy, xi, alpha):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.xi = xi
        self.alpha = alpha

    def double_sphere_project(self, points_3d):
        """双球模型投影（批量）"""
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        
        d1 = np.sqrt(x**2 + y**2 + z**2)
        k = self.xi * d1 + z
        d2 = np.sqrt(x**2 + y**2 + k**2)
        denom = self.alpha * d2 + (1 - self.alpha) * k
        
        # 防止除零或无效投影 (denom <= 0 表示超出有效视场)
        valid = denom > 0
        mx = np.zeros_like(denom)
        my = np.zeros_like(denom)
        mx[valid] = x[valid] / denom[valid]
        my[valid] = y[valid] / denom[valid]
        
        u = self.fx * mx + self.cx
        v = self.fy * my + self.cy
        return np.column_stack((u, v))
    
    def generate_pinhole_to_double_sphere_map(
        self,
        fx_pin, fy_pin, cx_pin, cy_pin,
        output_width, output_height
    ):
        """生成从虚拟针孔图像到原始双球图像的重映射表"""
        u_grid, v_grid = np.meshgrid(np.arange(output_width), np.arange(output_height))
        u_flat = u_grid.flatten().astype(np.float64)
        v_flat = v_grid.flatten().astype(np.float64)
        
        # 虚拟针孔：像素 → 归一化平面 → 3D方向向量
        mx = (u_flat - cx_pin) / fx_pin
        my = (v_flat - cy_pin) / fy_pin
        mz = np.ones_like(mx)
        norm = np.sqrt(mx*mx + my*my + mz*mz)
        rays = np.column_stack((mx/norm, my/norm, mz/norm))
        
        # 投影到双球模型
        pixels_ds = self.double_sphere_project(rays)
        
        map_x = pixels_ds[:, 0].reshape(output_height, output_width)
        map_y = pixels_ds[:, 1].reshape(output_height, output_width)
        
        # OpenCV remap 严格要求 float32
        return map_x.astype(np.float32), map_y.astype(np.float32)
    
    def remap_to_pinhole(
        self,
        image_ds,
        fx_pin, fy_pin, cx_pin, cy_pin, 
        output_size
    ):
        """将双球模型图像重映射到针孔模型图像"""
        output_width, output_height = output_size
        
        map_x, map_y = self.generate_pinhole_to_double_sphere_map(
            fx_pin, fy_pin, cx_pin, cy_pin,
            output_width, output_height
        )
        
        # 使用 BORDER_CONSTANT 填充无效/边缘区域为黑色 (0)
        output = cv2.remap(image_ds, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return output
