import argparse
import torch
import numpy as np
import warp as wp

import os
import json

# 1. 基础 AppLauncher 配置
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab AUV VLA 闭环控制系统 - 物理修正版")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 启动后导入 Isaac Lab 相关库 ---
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg
import omni.ui as ui

# 导入自定义工具类
from utils.camera import UnderwaterCameraManager, UnderwaterScene
from utils.llm import VLAController

# 初始化 Warp
wp.init()

# ----------------------------------------------------------------              
# 场景与资产配置
# ----------------------------------------------------------------
@configclass
class OceanSceneCfg(InteractiveSceneCfg):
    # 1. 静态海底环境 (使用 AssetBaseCfg)
    ocean_floor = AssetBaseCfg(
        prim_path="/World/OceanFloor",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/huohua/IsaacLab/ocean/final_floor/landing_floor.usd",
            scale=(1.0, 1.0, 1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(), # 开启碰撞
        )
    )

    # 2. 水下航行器 (AUV)
    auv = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/AUV",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/huohua/IsaacLab/ocean/auv/sea_auv.usd",
            # --- 核心强制修正 ---
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_depenetration_velocity=1.0,
                disable_gravity=True, # 模拟浮力中性
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0), # 必须给一个非零质量
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # 自动将该 Prim 标记为刚体
            func=sim_utils.spawners.from_files.spawn_from_usd, 
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0, 0, 2000),
            rot=(1.0,0,0,0)
        ),
    )

    # 3. 挂载摄像头 (直接作为 AUV 的子 Prim)
    cam_front = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_front",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(),
        offset=CameraCfg.OffsetCfg(pos=(0.5, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),convention='world')
    )

    # 后向摄像头 (Back) - 看向 X 轴负方向 (绕 Z 轴转 180度)
    cam_back = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_back",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(),
        offset=CameraCfg.OffsetCfg(pos=(-0.5, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0),convention='world')
    )

    # 左向摄像头 (Left) - 看向 Y 轴正方向 (绕 Z 轴转 90度)
    cam_left = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_left",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.5, 0.0), rot=(0.707, 0.0, 0.0, 0.707),convention='world')
    )

    # 右向摄像头 (Right) - 看向 Y 轴负方向 (绕 Z 轴转 -90度)
    cam_right = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_right",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(),
        offset=CameraCfg.OffsetCfg(pos=(0.0, -0.5, 0.0), rot=(0.707, 0.0, 0.0, -0.707),convention='world')
    )

    # 上向摄像头 (Up) - 看向 Z 轴正方向 (绕 Y 轴转 -90度)
    cam_up = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_up",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.5), rot=(0.707, 0.0, -0.707, 0.0),convention='world')
    )

    # 下向摄像头 (Down) - 看向 Z 轴负方向 (绕 Y 轴转 90度)
    # 这就是你原本的 uw_camera，统一命名为 cam_down
    cam_down = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_down",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, -0.5), rot=(0.707, 0.0, 0.707, 0.0),convention='world')
    )

    cam_follow = CameraCfg(
    prim_path="{ENV_REGEX_NS}/AUV/cam_follow",
    update_period=0, height=480, width=640, data_types=["rgb","distance_to_camera"],
    spawn=sim_utils.PinholeCameraCfg(),
    offset=CameraCfg.OffsetCfg(
        pos=(-20, 0.0, 20), 
        rot=(0.92388, 0.0, 0.38268, 0.0), # 绕Y轴旋转约45度视角
        convention='world'
    )
)
# ----------------------------------------------------------------              
# 主程序
# ----------------------------------------------------------------
def main():
    # 1. 仿真环境初始化 (保持之前的 OceanSceneCfg 不变)
    sim_context = SimulationContext(sim_utils.SimulationCfg(dt=0.02, device="cuda:0"))
    scene = InteractiveScene(OceanSceneCfg(num_envs=1, env_spacing=0.0))
    
    # 2. 获取所有摄像头引用
    # 这里的 key 必须与 OceanSceneCfg 中定义的变量名一致
    camera_names = ["cam_front", "cam_back", "cam_left", "cam_right", "cam_up", "cam_down"]
    camera_sensors = {name: scene[name] for name in camera_names}
    
    auv_robot: RigidObject = scene["auv"]
    
    window = ui.Window("Underwater View (Processed)", width=1280, height=720)
    provider = ui.ByteImageProvider()
    with window.frame:
        ui.ImageWithProvider(provider)

    # 3. 工具类初始化 - 升级后的 Manager 现在接受传感器字典
    cam_manager = UnderwaterCameraManager(camera_sensors)
    monitor_cam_manager = UnderwaterScene(scene["cam_follow"], device="cuda:0")
    vla_client = VLAController(api_key="")
    
    sim_context.reset()
    current_vel = torch.zeros((1, 6), device="cuda:0")
    step_count = 0

    while simulation_app.is_running():
        if sim_context.is_playing():
            auv_robot.write_root_velocity_to_sim(current_vel)
            processed_tensor = monitor_cam_manager.process_frame()
        
            if processed_tensor is not None:
                # 这里的 provider 是你定义的 ui.ByteImageProvider
                # 使用 data_ptr() 高效传输 GPU 内存地址
                provider.set_bytes_data_from_gpu(
                    processed_tensor.data_ptr(), 
                    (monitor_cam_manager.width, monitor_cam_manager.height)
                )
            
        sim_context.step(render=True)
        
        if sim_context.is_playing():
            if step_count % 100 == 0:
                # 获取 6 个方向处理后的图片路径列表
                img_paths = cam_manager.capture_all_and_process()
                
                if img_paths:
                    # 这里的 task 应当引导 LLM 综合观察
                    task = """
**[Your Task]**
As the Control Expert, based on your analysis of the provided real-time images, execute your decision cycle according to the following strict protocol:
1.  **PRIMARY ANALYSIS - TARGET ACQUISITION:** Scrutinize the six (6) non-target, real-time camera feeds provided. Is the target object visible in any of them?
2.  **DECISION & ACTION:**
    -   **IF TARGET IS VISIBLE:** You must immediately initiate the 'Target ' protocol. Report using the "$$...$$" format and issue commands to approach and position over the target.
            command is one of [ `ascend`, `descend`, `move left`, `move right`, `move forward`, `move backward`, `stop`]
"""
                    try:
                        # 1. 获取 LLM 响应
                        response = vla_client.get_action(img_paths, task)
                        print(f"VLA Response: {response}")

                        # 2. 初始化速度向量 [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
                        # Isaac Lab 中，通常 X 正向为前，Y 正向为左，Z 正向为上
                        new_vel = np.zeros(6)
                        speed_scale = 30.0  # 设定一个基础速度值 (m/s)

                        # 3. 简单的文本解析逻辑 (假设 response 包含指令关键字)
                        res_lower = response.lower()
                        
                        if "move forward" in res_lower:
                            new_vel[0] = speed_scale
                        elif "move backward" in res_lower:
                            new_vel[0] = -speed_scale
                        
                        if "move left" in res_lower:
                            new_vel[1] = speed_scale
                        elif "move right" in res_lower:
                            new_vel[1] = -speed_scale
                        
                        if "ascend" in res_lower:
                            new_vel[2] = speed_scale
                        elif "descend" in res_lower:
                            new_vel[2] = -speed_scale
                            
                        if "stop" in res_lower:
                            new_vel = np.zeros(6)

                        # 4. 更新当前速度 Tensor 并应用到仿真
                        current_vel = torch.tensor([new_vel], device="cuda:0", dtype=torch.float)
                        auv_robot.write_root_velocity_to_sim(current_vel)
                        print(f"Applying Velocity: {new_vel}")

                    except Exception as e:
                        print(f"Decision Error: {e}")
            
            # 持续应用速度 (确保 AUV 在 step 之间保持运动)
            
            
            
            step_count += 1
    simulation_app.close()

if __name__ == "__main__":
    main()