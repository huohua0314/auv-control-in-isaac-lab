import argparse
import torch
import numpy as np
import warp as wp
import random
import os
import json

# 1. basic AppLauncher assest
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Lab AUV VLA 闭环控制系统 - 物理修正版")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---  Isaac Lab Related Libraries ---
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg
import omni.ui as ui

# Import custom utility classes
from utils.camera import UnderwaterCameraManager, UnderwaterScene
from utils.llm import VLAController
from utils.asycn import AsyncAUVController
from utils.prompt.task2 import formatted_prompt
# Initialize Warp
def get_single_random_auv_pos():
    """
    生成单个 AUV 的随机初始坐标 (单位: cm)
    确保返回的是纯 float 元组，适配 RigidObjectCfg.InitialStateCfg
    """
    # set range 
    x_range = (-500.0, 500.0)
    y_range = (-500.0, 500.0)
    z_range = (1300.0, 1800.0)
    
    # random
    x = float(random.uniform(*x_range))
    y = float(random.uniform(*y_range))
    z = float(random.uniform(*z_range))
    
  
    return (x, y, z)


wp.init()

@configclass
class OceanSceneCfg(InteractiveSceneCfg):
    # Oceanscene
    ocean_floor = AssetBaseCfg(
        prim_path="/World/OceanFloor",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/huohua/IsaacLab/ocean/final_floor/fordock.usd",
            scale=(1.0, 1.0, 1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(), # Collision

        )
    )

    # (AUV)
    auv = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/AUV",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/huohua/IsaacLab/ocean/mesh/auv.usd",
            # --- 核心强制修正 ---
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_depenetration_velocity=1.0,
                disable_gravity=True, 
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0), 
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # 自动将该 Prim 标记为刚体
            func=sim_utils.spawners.from_files.spawn_from_usd, 
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=get_single_random_auv_pos(),
            rot=(1.0,0,0,0),
        ),
    )

    # 3. camera
    cam_front = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_front",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(50,1000000),focal_length=12.0),
        offset=CameraCfg.OffsetCfg(pos=(1, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),convention='world')
    )

    # Back camera 
    cam_back = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_back",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(50,1000000),focal_length=12.0),
        offset=CameraCfg.OffsetCfg(pos=(-1, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0),convention='world')
    )

    # left camera
    cam_left = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_left",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(25,1000000),focal_length=12.0),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 1, 0.0), rot=(0.707, 0.0, 0.0, 0.707),convention='world')
    )

    # right camera
    cam_right = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_right",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(25,1000000),focal_length=12.0),
        offset=CameraCfg.OffsetCfg(pos=(0.0, -1, 0.0), rot=(0.707, 0.0, 0.0, -0.707),convention='world')
    )

    # up camera
    cam_up = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_up",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
       spawn=sim_utils.PinholeCameraCfg(clipping_range=(25,1000000),focal_length=12.0),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 1), rot=(0.707, 0.0, -0.707, 0.0),convention='world')
    )

    # down camera
   
    cam_down = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AUV/cam_down",
        update_period=0, height=240, width=320, data_types=["rgb","distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(25,1000000),focal_length=12.0),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, -100), rot=(0.707, 0.0, 0.707, 0.0),convention='world')
    )

    # view camera
    cam_follow = CameraCfg(
    prim_path="{ENV_REGEX_NS}/AUV/cam_follow",
    update_period=0, height=480, width=640, data_types=["rgb","distance_to_camera"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=12.0),
    offset=CameraCfg.OffsetCfg(
        pos=(-500, 0.0, 500), 
        rot=(0.92388, 0.0, 0.38268, 0.0), 
        convention='world'
    )
)
# ----------------------------------------------------------------              
# main 
# ----------------------------------------------------------------
def main():
    # 1. scene initial
    sim_context = SimulationContext(sim_utils.SimulationCfg(dt=0.02, device="cuda:0"))
    scene = InteractiveScene(OceanSceneCfg(num_envs=1, env_spacing=0.0))
    
    # 2. create camera
    camera_names = ["cam_front", "cam_back", "cam_left", "cam_right", "cam_up", "cam_down"]
    camera_sensors = {name: scene[name] for name in camera_names}
    
    auv_robot: RigidObject = scene["auv"]
    
    window = ui.Window("Underwater View (Processed)", width=1280, height=720)
    provider = ui.ByteImageProvider()
    with window.frame:
        ui.ImageWithProvider(provider)

    # 3. tools inital
    cam_manager = UnderwaterCameraManager(camera_sensors)
    monitor_cam_manager = UnderwaterScene(scene["cam_follow"], device="cuda:0")
    vla_client = VLAController(api_key="")
    
    async_ctrl = AsyncAUVController(vla_client, speed_scale=10.0)

    sim_context.reset()
    current_vel = torch.zeros((1, 6), device="cuda:0")
    step_count = 0

    while simulation_app.is_running():
        if sim_context.is_playing():
            auv_robot.write_root_velocity_to_sim(current_vel)
            processed_tensor = monitor_cam_manager.process_frame()
        
            if processed_tensor is not None:
               
                provider.set_bytes_data_from_gpu(
                    processed_tensor.data_ptr(), 
                    (monitor_cam_manager.width, monitor_cam_manager.height)
                )
            
        sim_context.step(render=True)
        
        if sim_context.is_playing():
            if step_count % 100 == 0:
                
                img_paths = cam_manager.capture_all_and_process()
                
                if img_paths:
                    #
                    task = """
**[Your Task]**
As the Control Expert, based on your analysis of the provided real-time images, execute your decision cycle according to the following strict protocol:
1.  **PRIMARY ANALYSIS - TARGET ACQUISITION:** Scrutinize the six (6) non-target, real-time camera feeds provided. Is the target object visible in any of them?
2.  **DECISION & ACTION:**
    -   **IF TARGET IS VISIBLE:** You must immediately initiate the 'Target ' protocol. Report using the "$$...$$" format and issue commands to approach and position over the target.
            command is one of [ `ascend`, `descend`, `move left`, `move right`, `move forward`, `move backward`, `stop`]
"""
                            
                    prompt = formatted_prompt + "\n" + task
                    async_ctrl.step_request(img_paths,prompt)

                    async_ctrl.refresh_state()

                    current_vel = async_ctrl.get_velocity_tensor()
                    auv_robot.write_root_velocity_to_sim(current_vel)
       
            step_count += 1
    simulation_app.close()

if __name__ == "__main__":
    main()