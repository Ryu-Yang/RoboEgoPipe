"""
双目相机校正与深度估计测试脚本

功能：
1. 使用已知左右相机内外参进行双目校正
2. 生成校正映射并对图像进行重映射
3. 输出校正后的相机参数
4. 计算视差图和深度图
5. 可视化校正效果（极线检查）
6. 支持手动微调立体校准参数
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import os
import argparse

# from roboegopipe.cammodel.double_sphere import DoubleSphereCameraModel


def create_rotation_x(angle_degrees):
    """
    创建绕X轴的旋转矩阵（垂直方向的旋转，即pitch调整）
    
    Args:
        angle_degrees: 旋转角度（度），正值为向下旋转，负值为向上旋转
    
    Returns:
        3x3 旋转矩阵
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R_x = np.array([
        [1, 0, 0],
        [0, cos_a, -sin_a],
        [0, sin_a, cos_a]
    ], dtype=np.float64)
    return R_x


def create_rotation_y(angle_degrees):
    """
    创建绕Y轴的旋转矩阵（水平方向的旋转，即yaw调整）
    
    Args:
        angle_degrees: 旋转角度（度），正值为向右旋转，负值为向左旋转
    
    Returns:
        3x3 旋转矩阵
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R_y = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ], dtype=np.float64)
    return R_y


def create_rotation_z(angle_degrees):
    """
    创建绕Z轴的旋转矩阵（滚转调整，即roll调整）
    
    Args:
        angle_degrees: 旋转角度（度），正值为顺时针旋转，负值为逆时针旋转
    
    Returns:
        3x3 旋转矩阵
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R_z = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    return R_z


def apply_manual_adjustment(R_cam1_from_cam2, t_cam1_from_cam2, 
                            roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
    """
    手动调整相对旋转矩阵
    
    Args:
        R_cam1_from_cam2: 原始相对旋转矩阵 (3x3)
        t_cam1_from_cam2: 原始平移向量 (3x1)
        roll_deg: 绕Z轴旋转角度（度）
        pitch_deg: 绕X轴旋转角度（度）- 垂直方向调整
        yaw_deg: 绕Y轴旋转角度（度）- 水平方向调整
    
    Returns:
        调整后的旋转矩阵和平移向量
    """
    # 创建各轴旋转矩阵
    R_x = create_rotation_x(pitch_deg)
    R_y = create_rotation_y(yaw_deg)
    R_z = create_rotation_z(roll_deg)
    
    # 组合旋转：R = R_z @ R_y @ R_x
    R_adjust = R_z @ R_y @ R_x
    
    # 应用调整：R_new = R_adjust @ R_original
    R_adjusted = R_adjust @ R_cam1_from_cam2
    
    print(f"\n🔧 手动调整参数:")
    print(f"  Roll (绕Z轴): {roll_deg:.4f}°")
    print(f"  Pitch (绕X轴/垂直): {pitch_deg:.4f}°")
    print(f"  Yaw (绕Y轴/水平): {yaw_deg:.4f}°")
    print(f"  调整旋转矩阵:\n{R_adjust}")
    
    return R_adjusted, t_cam1_from_cam2


def parse_tvec_quat(params):
    """
    从 7 维数组解析外参
    输入格式: [tx, ty, tz, qx, qy, qz, qw]
    返回: 旋转矩阵 (3x3), 平移向量 (3x1)
    """
    t = params[:3].reshape(3, 1)
    # scipy 期望四元数顺序为 [qx, qy, qz, qw]
    quat_scipy = np.array([params[3], params[4], params[5], params[6]])
    R = Rotation.from_quat(quat_scipy).as_matrix()
    return R, t


def compute_depth_from_disparity(disparity, Q):
    """
    使用 Q 矩阵将视差图转换为深度图
    
    Args:
        disparity: 视差图 (float32)
        Q: 重投影矩阵 (4x4)
    
    Returns:
        depth: 深度图 (float32)，单位：米
    """
    depth = np.zeros_like(disparity)
    valid_mask = disparity > 0
    
    if np.any(valid_mask):
        # 深度公式: depth = Q[2,3] / (disparity * (-Q[3,2]))
        depth[valid_mask] = Q[2, 3] / (disparity[valid_mask] * (-Q[3, 2]))
    
    # 处理无效值
    invalid_mask = ~np.isfinite(depth)
    depth[invalid_mask] = 0.0
    
    # 过滤异常值
    max_depth = 20.0
    depth[(depth > max_depth) | (depth < 0)] = 0.0
    
    return depth


def draw_epipolar_lines(img_l, img_r, P1, P2, num_lines=10):
    """
    在立体校正后的图像上绘制极线，用于验证校正效果
    
    Args:
        img_l, img_r: 左右校正图像
        P1, P2: 投影矩阵
        num_lines: 绘制的极线数量
    
    Returns:
        带有极线的左右图像
    """
    h, w = img_l.shape[:2]
    img_l_lines = img_l.copy()
    img_r_lines = img_r.copy()
    
    # 在左图像上均匀选择点，绘制对应的极线（水平线）
    step = h // num_lines
    for i in range(0, num_lines):
        y = step * i + step // 2
        if y >= h:
            break
        color = (0, 255, 0) if i % 2 == 0 else (0, 0, 255)
        cv2.line(img_l_lines, (0, y), (w - 1, y), color, 1)
        cv2.line(img_r_lines, (0, y), (w - 1, y), color, 1)
    
    return img_l_lines, img_r_lines


def visualize_disparity(disparity):
    """
    将视差图转换为可视化的彩色图像
    
    Args:
        disparity: 视差图
    
    Returns:
        可视化的视差彩色图
    """
    disp_valid = disparity[disparity > 0]
    if len(disp_valid) == 0:
        return np.zeros_like(disparity, dtype=np.uint8)
    
    disp_min = disp_valid.min()
    disp_max = disp_valid.max()
    
    # 归一化到 0-255
    disp_vis = ((disparity - disp_min) / (disp_max - disp_min) * 255).astype(np.uint8)
    disp_vis[disparity <= 0] = 0
    
    # 转换为彩色图
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
    return disp_color


if __name__ == "__main__":
    # ================= 解析命令行参数 =================
    parser = argparse.ArgumentParser(description="双目相机校正与深度估计（支持手动微调）")
    parser.add_argument("--roll", type=float, default=0.0,
                        help="绕Z轴旋转角度（度），用于滚转调整")
    parser.add_argument("--pitch", type=float, default=0.0,
                        help="绕X轴旋转角度（度），用于垂直方向调整（pitch）")
    parser.add_argument("--yaw", type=float, default=0.0,
                        help="绕Y轴旋转角度（度），用于水平方向调整（yaw）")
    parser.add_argument("--interactive", action="store_true",
                        help="启用交互模式，逐步调整参数")
    args = parser.parse_args()
    
    # 打印使用说明
    if args.interactive:
        print("=" * 60)
        print("🎮 交互模式使用说明")
        print("=" * 60)
        print("在交互模式下，您可以逐步调整校准参数：")
        print("  - 按 'q'/'a': 增加/减少 Pitch (垂直方向)")
        print("  - 按 'w'/'s': 增加/减少 Yaw (水平方向)")
        print("  - 按 'e'/'d': 增加/减少 Roll (滚转)")
        print("  - 按 'r': 重置所有参数")
        print("  - 按 'Enter': 使用当前参数继续处理")
        print("  - 按 'Esc': 退出")
        print("=" * 60)

    # 创建输出目录
    os.makedirs("output/rectify_images", exist_ok=True)
    
    # ================= 1. 构建左相机内参矩阵 =================
    fx_l, fy_l = 507.87387315510574, 509.4144511814894
    cx_l, cy_l = 796.7150518282281, 655.6687369325308
    K1 = np.array([[fx_l, 0, cx_l],
                   [0, fy_l, cy_l],
                   [0, 0, 1]], dtype=np.float64)

    # ================= 2. 构建右相机内参矩阵 =================
    fx_r, fy_r = 511.053073270337, 512.7060865113444
    cx_r, cy_r = 804.5116352623274, 617.9051244593389
    K2 = np.array([[fx_r, 0, cx_r],
                   [0, fy_r, cy_r],
                   [0, 0, 1]], dtype=np.float64)

    # ================= 3. 解析外参 =================
    # T_target_from_source: 将 source 系点变换到 target 系
    # 相机到body的变换矩阵。7 维数组格式为: [tx, ty, tz, qx, qy, qz，qw].
    T_body_from_cam1 = np.array([0.03217024, -0.0023996, -0.00579464, 
                                 -0.00006496899371050576, 0.9999770106002994, 
                                 0.0007486897132216666, 0.006738954936181263], dtype=np.float64)
    T_body_from_cam2 = np.array([-0.02795044, -0.00278565, -0.00521396, 
                                 -0.006224006661330271, 0.9999716180538576, 
                                 -0.003033076090405442, -0.0029707368209763796], dtype=np.float64)

    R_body_from_cam1, t_body_from_cam1 = parse_tvec_quat(T_body_from_cam1)
    R_body_from_cam2, t_body_from_cam2 = parse_tvec_quat(T_body_from_cam2)

    # 相对外参: T_cam1_from_cam2 (将 cam2 系点变换到 cam1 系)
    # T_cam1_from_cam2 = T_cam1_from_body @ T_body_from_cam2
    #                  = T_body_from_cam1.inverse() @ T_body_from_cam2
    R_cam1_from_cam2 = R_body_from_cam1.T @ R_body_from_cam2
    t_cam1_from_cam2 = R_body_from_cam1.T @ (t_body_from_cam2 - t_body_from_cam1)
    
    # ================= 4.5 手动调整相对外参 =================
    print("\n" + "=" * 60)
    print("📐 原始相对外参 (Cam2->Cam1):")
    print("=" * 60)
    print(f"  R_cam1_from_cam2:\n{R_cam1_from_cam2}")
    print(f"  t_cam1_from_cam2: {t_cam1_from_cam2.flatten()}")
    
    # 存储原始值用于交互模式
    R_cam1_from_cam2_original = R_cam1_from_cam2.copy()
    t_cam1_from_cam2_original = t_cam1_from_cam2.copy()
    
    # ================= 5. 设置畸变参数 =================
    # 注意：OpenCV 的 stereoRectify 默认支持针孔+多项式畸变模型。
    # 若实际使用"双球模型(双鱼眼/全向)"，需改用 cv2.omnidir 模块或先将参数投影至针孔模型。
    D1 = np.zeros((5, 1), dtype=np.float64)  # [k1, k2, p1, p2, k3]
    D2 = np.zeros((5, 1), dtype=np.float64)

    image_size = (1600, 1300)  # (width, height)

    # 加载图像用于交互模式
    image_l = cv2.imread("output/undistorted_images/l_pinhole_output.jpg")
    image_r = cv2.imread("output/undistorted_images/r_pinhole_output.jpg")
    
    # 交互模式
    if args.interactive:
        if image_l is None or image_r is None:
            raise FileNotFoundError("交互模式需要输入图像: output/undistorted_images/l_pinhole_output.jpg 和 r_pinhole_output.jpg")
        
        current_roll = 0.0
        current_pitch = 0.0
        current_yaw = 0.0
        step_size = 0.1  # 默认步长
        
        print("进入交互模式，调整参数观察校正效果...")
        
        while True:
            # 应用当前调整
            R_cam1_from_cam2_adj, t_cam1_from_cam2_adj = apply_manual_adjustment(
                R_cam1_from_cam2_original, t_cam1_from_cam2_original,
                current_roll, current_pitch, current_yaw
            )
            
            # 执行双目校正并显示结果
            try:
                R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                    K1, D1, K2, D2, image_size, R_cam1_from_cam2_adj, t_cam1_from_cam2_adj, alpha=0
                )
                
                # 生成校正映射
                l_map_x, l_map_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
                r_map_x, r_map_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
                
                image_rect_l = cv2.remap(image_l, l_map_x, l_map_y, cv2.INTER_LINEAR)
                image_rect_r = cv2.remap(image_r, r_map_x, r_map_y, cv2.INTER_LINEAR)
                
                # 绘制极线检查
                img_l_lines, img_r_lines = draw_epipolar_lines(image_rect_l, image_rect_r, P1, P2)
                combined = np.hstack([img_l_lines, img_r_lines])
                
                # 在图像上显示当前参数
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(combined, f"Roll:{current_roll:+.2f} Pitch:{current_pitch:+.2f} Yaw:{current_yaw:+.2f}",
                           (10, 30), font, 0.8, (0, 255, 0), 2)
                cv2.putText(combined, f"Step:{step_size:.2f} (use +/- to change)",
                           (10, 60), font, 0.6, (255, 255, 0), 1)
                
                cv2.imshow("Interactive Rectification Adjustment (Press ESC to exit)", combined)
                key = cv2.waitKey(0) & 0xFF
                
                if key == 27:  # ESC
                    print("\n退出交互模式")
                    break
                elif key == ord('q'):
                    current_pitch += step_size
                elif key == ord('a'):
                    current_pitch -= step_size
                elif key == ord('w'):
                    current_yaw += step_size
                elif key == ord('s'):
                    current_yaw -= step_size
                elif key == ord('e'):
                    current_roll += step_size
                elif key == ord('d'):
                    current_roll -= step_size
                elif key == ord('r'):
                    current_roll = 0.0
                    current_pitch = 0.0
                    current_yaw = 0.0
                    print("参数已重置")
                elif key == ord('+') or key == ord('='):
                    step_size *= 2
                    print(f"步长增加到: {step_size}")
                elif key == ord('-'):
                    step_size /= 2
                    print(f"步长减少到: {step_size}")
                elif key == 13 or key == 32:  # Enter or Space
                    print(f"使用当前参数: Roll={current_roll}, Pitch={current_pitch}, Yaw={current_yaw}")
                    # 应用最终参数
                    R_cam1_from_cam2, t_cam1_from_cam2 = apply_manual_adjustment(
                        R_cam1_from_cam2_original, t_cam1_from_cam2_original,
                        current_roll, current_pitch, current_yaw
                    )
                    args.roll = current_roll
                    args.pitch = current_pitch
                    args.yaw = current_yaw
                    break
                    
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"校正失败: {e}")
                break
    else:
        # 使用命令行参数
        if args.roll != 0.0 or args.pitch != 0.0 or args.yaw != 0.0:
            R_cam1_from_cam2, t_cam1_from_cam2 = apply_manual_adjustment(
                R_cam1_from_cam2_original, t_cam1_from_cam2_original,
                args.roll, args.pitch, args.yaw
            )
    
    print("\n" + "=" * 60)
    print("📐 最终使用的相对外参 (Cam2->Cam1):")
    print("=" * 60)
    print("📷 双目相机参数")
    print("=" * 60)
    print(f"左相机内参矩阵 K1:\n{K1}")
    print(f"右相机内参矩阵 K2:\n{K2}")
    print(f"\n左相机外参 (Cam1->Body):")
    print(f"  R_body_from_cam1:\n{R_body_from_cam1}")
    print(f"  t_body_from_cam1: {t_body_from_cam1.flatten()}")
    print(f"\n右相机外参 (Cam2->Body):")
    print(f"  R_body_from_cam2:\n{R_body_from_cam2}")
    print(f"  t_body_from_cam2: {t_body_from_cam2.flatten()}")
    print(f"\n相对外参 (Cam2->Cam1):")
    print(f"  R_cam1_from_cam2:\n{R_cam1_from_cam2}")
    print(f"  t_cam1_from_cam2: {t_cam1_from_cam2.flatten()}")
    
    # 计算基线长度
    baseline = np.linalg.norm(t_cam1_from_cam2)
    print(f"\n✅ 基线长度: {baseline * 1000:.2f} mm")

    # ================= 6. 双目校正 =================
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R_cam1_from_cam2, t_cam1_from_cam2, alpha=0
    )

    print("\n" + "=" * 60)
    print("✅ 双目校正完成")
    print("=" * 60)
    print(f"左相机校正旋转矩阵 R1:\n{R1}")
    print(f"\n右相机校正旋转矩阵 R2:\n{R2}")
    print(f"\n左相机投影矩阵 P1:\n{P1}")
    print(f"\n右相机投影矩阵 P2:\n{P2}")
    print(f"\n立体视差-深度转换矩阵 Q:\n{Q}")
    print(f"\n有效图像区域 ROI1: {validPixROI1}")
    print(f"有效图像区域 ROI2: {validPixROI2}")

    # ================= 7. 生成校正映射并应用 =================
    # 图像已在前面加载，检查是否有效
    if image_l is None or image_r is None:
        raise FileNotFoundError("请提供有效的输入图像路径: output/undistorted_images/l_pinhole_output.jpg 和 r_pinhole_output.jpg")

    l_map_x, l_map_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    r_map_x, r_map_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    image_rect_l = cv2.remap(image_l, l_map_x, l_map_y, cv2.INTER_LINEAR)
    image_rect_r = cv2.remap(image_r, r_map_x, r_map_y, cv2.INTER_LINEAR)

    cv2.imwrite("output/rectify_images/l_rect_output.jpg", image_rect_l)
    cv2.imwrite("output/rectify_images/r_rect_output.jpg", image_rect_r)
    
    print(f"\n✅ 校正图像已保存:")
    print(f"  output/rectify_images/l_rect_output.jpg")
    print(f"  output/rectify_images/r_rect_output.jpg")

    # ================= 8. 输出左相机新的内外参 =================
    # 校正后的左相机内参（从投影矩阵 P1 提取）
    K1_rect = P1[:3, :3]
    # 校正后的左相机外参：旋转矩阵为 R1，平移为零（因为校正后左相机作为参考坐标系）
    R1_new = R1
    t1_new = np.zeros((3, 1))
    
    # 右相机校正后的外参（相对于左相机）
    R2_new = R2 @ R_cam1_from_cam2
    t2_new = R2 @ R_cam1_from_cam2.T @ t_cam1_from_cam2 + t_body_from_cam2 - t_body_from_cam1
    
    print("\n" + "=" * 60)
    print("📐 校正后相机参数（用于后续深度估计）")
    print("=" * 60)
    print(f"左相机新内参矩阵 K1_rect:\n{K1_rect}")
    print(f"左相机新旋转矩阵 R1:\n{R1_new}")
    print(f"左相机新平移向量 t1: [0, 0, 0] (参考坐标系)")
    print(f"\n右相机新内参矩阵 K2_rect:\n{P2[:3, :3]}")
    print(f"右相机新旋转矩阵 R2:\n{R2}")
    print(f"\n重投影矩阵 Q:\n{Q}")
    
    # 提取深度估计关键参数
    focal_length_rect = (K1_rect[0, 0] + K1_rect[1, 1]) / 2  # 平均焦距
    baseline_rect = -1.0 / Q[3, 2]  # 从 Q 矩阵提取的基线
    print(f"\n🔑 深度估计关键参数:")
    print(f"  校正后焦距 f: {focal_length_rect:.4f}")
    print(f"  基线长度 baseline: {baseline_rect * 1000:.2f} mm")
    print(f"  深度公式: depth = (f * baseline) / disparity")
    print(f"  depth (m) = {focal_length_rect * baseline_rect:.4f} / disparity")

    # ================= 9. 计算视差图和深度图 =================
    print("\n" + "=" * 60)
    print("🔍 计算视差图和深度图")
    print("=" * 60)
    
    # 转换为灰度图用于立体匹配
    gray_l = cv2.cvtColor(image_rect_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(image_rect_r, cv2.COLOR_BGR2GRAY)
    
    # 创建 SGBM 立体匹配器
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=11,
        P1=8 * 3 * 11 * 11,
        P2=32 * 3 * 11 * 11,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # 计算视差
    disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
    
    # 计算深度
    depth = compute_depth_from_disparity(disparity, Q)
    
    # 保存视差图和深度图
    np.save("output/rectify_images/disparity.npy", disparity)
    np.save("output/rectify_images/depth.npy", depth)
    
    # 可视化视差图
    disp_vis = visualize_disparity(disparity)
    cv2.imwrite("output/rectify_images/disparity_vis.png", disp_vis)
    
    # 统计信息
    disp_valid = disparity[disparity > 0]
    depth_valid = depth[depth > 0]
    
    print(f"视差统计:")
    if len(disp_valid) > 0:
        print(f"  最小视差: {disp_valid.min():.2f}")
        print(f"  最大视差: {disp_valid.max():.2f}")
        print(f"  平均视差: {disp_valid.mean():.2f}")
        print(f"  有效像素: {len(disp_valid)}/{disparity.size} ({100*len(disp_valid)/disparity.size:.1f}%)")
    else:
        print("  无有效视差值")
    
    print(f"\n深度统计:")
    if len(depth_valid) > 0:
        print(f"  最小深度: {depth_valid.min():.3f} m")
        print(f"  最大深度: {depth_valid.max():.3f} m")
        print(f"  平均深度: {depth_valid.mean():.3f} m")
        print(f"  有效像素: {len(depth_valid)}/{depth.size} ({100*len(depth_valid)/depth.size:.1f}%)")
    else:
        print("  无有效深度值")
    
    print(f"\n✅ 视差图和深度图已保存:")
    print(f"  output/rectify_images/disparity.npy (原始视差)")
    print(f"  output/rectify_images/depth.npy (原始深度)")
    print(f"  output/rectify_images/disparity_vis.png (视差可视化)")

    # ================= 10. 极线检查可视化 =================
    print("\n" + "=" * 60)
    print("📊 极线检查（验证校正效果）")
    print("=" * 60)
    
    img_l_lines, img_r_lines = draw_epipolar_lines(image_rect_l, image_rect_r, P1, P2)
    cv2.imwrite("output/rectify_images/epipolar_check.png", 
                np.hstack([img_l_lines, img_r_lines]))
    print("✅ 极线检查图像已保存: output/rectify_images/epipolar_check.png")
    print("   (绿色/红色水平线应穿过左右图像的对应点)")

    print("\n" + "=" * 60)
    print("🎉 所有处理完成！")
    print("=" * 60)