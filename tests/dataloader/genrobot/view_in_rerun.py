from pathlib import Path

from roboegopipe.dataloader.genrobot import GenrobotdataLoader
from roboegopipe.viewer.traj import visualize_trajectory_with_rerun


def main():
    """主函数"""
    # 替换为你的实际路径
    mcap_path = "/home/ryu-yang/Documents/Datasets/Domestic_Services/Living_Room/Organization/Organize_desktop/3a8f559dfb0847c8be710fa31c37758a.mcap"
    
    # 检查文件是否存在
    if not Path(mcap_path).exists():
        print(f"❌ 文件不存在: {mcap_path}")
        print("请修改代码中的 mcap_path 变量为有效的MCAP文件路径")
        return
    
    dataLoader = GenrobotdataLoader(mcap_path)
    dataLoader.read_data()
    traj = dataLoader.get_traj()
    visualize_trajectory_with_rerun(traj)

if __name__ == "__main__":
    main()
