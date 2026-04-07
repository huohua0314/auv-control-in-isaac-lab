import os
import torch
import numpy as np
import warp as wp
import PIL.Image as Image
from isaaclab.sensors import Camera, CameraCfg

# --- Warp 核函数：处理水下衰减与散射 ---
@wp.kernel
def UW_render_kernel(
    raw_rgba: wp.array(dtype=wp.float32, ndim=3),
    depth: wp.array(dtype=wp.float32, ndim=2),
    backscatter_value: wp.vec3,
    atten_coeff: wp.vec3,
    backscatter_coeff: wp.vec3,
    uw_image: wp.array(dtype=wp.float32, ndim=3)
):
    i, j = wp.tid()
    # 深度转换系数（根据场景尺度调整）
    d = depth[i, j] * 0.0005
    
    # 透射率与散射增益
    trans = wp.vec3(wp.exp(-atten_coeff[0] * d), wp.exp(-atten_coeff[1] * d), wp.exp(-atten_coeff[2] * d))
    sc = wp.vec3(1.0 - wp.exp(-backscatter_coeff[0] * d), 1.0 - wp.exp(-backscatter_coeff[1] * d), 1.0 - wp.exp(-backscatter_coeff[2] * d))
    
    # 合成最终颜色
    uw_image[i, j, 0] = raw_rgba[i, j, 0] * trans[0] + backscatter_value[0] * sc[0]
    uw_image[i, j, 1] = raw_rgba[i, j, 1] * trans[1] + backscatter_value[1] * sc[1]
    uw_image[i, j, 2] = raw_rgba[i, j, 2] * trans[2] + backscatter_value[2] * sc[2]
    uw_image[i, j, 3] = 1.0

class UnderwaterCameraManager:
    def __init__(self, camera_sensors: dict, device: str = "cuda:0"):
        """
        camera_sensors: 格式为 {"cam_front": sensor_object, ...}
        """
        self.sensors = camera_sensors
        self.device = device
        
        # 假设所有摄像头分辨率一致，取第一个作为标准
        first_sensor = next(iter(camera_sensors.values()))
        self.height = first_sensor.cfg.height
        self.width = first_sensor.cfg.width
        
        # 水下参数 (保持不变)
        self.backscatter_val = wp.vec3(0.02, 0.08, 0.15)
        self.atten_coeff = wp.vec3(0.8, 0.3, 0.15) 
        self.backscatter_coeff = wp.vec3(0.05, 0.05, 0.05)
        
        # 预分配输出显存
        self.uw_output_wp = wp.zeros((self.height, self.width, 4), dtype=wp.float32, device=self.device)
        
        self.temp_dir = "temp_obs"
        os.makedirs(self.temp_dir, exist_ok=True)

    def capture_all_and_process(self) -> list:
        """
        处理所有挂载的摄像头，返回本地路径列表。
        顺序：Front, Back, Left, Right, Up, Down (取决于 main 中的 camera_names)
        """
        saved_paths = []
        
        for name, sensor in self.sensors.items():
            # 1. 更新单个传感器
            sensor.update(dt=0.01)
            
            if "rgb" not in sensor.data.output or "distance_to_camera" not in sensor.data.output:
                continue

            # 2. 获取数据并转为 Warp 格式
            rgb_raw = sensor.data.output["rgb"][0].float().contiguous() / 255.0
            depth_raw = sensor.data.output["distance_to_camera"][0].float().contiguous()
            if depth_raw.dim() == 3: depth_raw = depth_raw.squeeze(-1)

            wp_rgb = wp.from_torch(rgb_raw, dtype=wp.float32)
            wp_depth = wp.from_torch(depth_raw, dtype=wp.float32)

            # 3. 运行 Warp 核函数
            wp.launch(
                kernel=UW_render_kernel,
                dim=(self.height, self.width),
                inputs=[wp_rgb, wp_depth, self.backscatter_val, self.atten_coeff, self.backscatter_coeff],
                outputs=[self.uw_output_wp],
                device=self.device
            )

            # 4. 保存为独立文件
            uw_torch = wp.to_torch(self.uw_output_wp)
            uw_uint8 = (torch.clamp(uw_torch[:, :, :3], 0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
            
            save_path = os.path.join(self.temp_dir, f"{name}.jpg")
            Image.fromarray(uw_uint8).save(save_path)
            saved_paths.append(save_path)
            
        return saved_paths


class UnderwaterScene:
    def __init__(self, camera_sensor, device: str = "cuda:0"):
        """
        camera_sensor: 单个 Camera 传感器对象
        """
        self.sensor = camera_sensor
        self.device = device
        
        # 获取分辨率
        self.height = camera_sensor.cfg.height
        self.width = camera_sensor.cfg.width
        
        # 水下光学参数 (可根据深度动态调整)
        self.backscatter_val = wp.vec3(0.02, 0.08, 0.15)
        self.atten_coeff = wp.vec3(0.8, 0.3, 0.15) 
        self.backscatter_coeff = wp.vec3(0.05, 0.05, 0.05)
        
        # 预分配输出显存 (使用 Torch 创建，再包装给 Warp，确保生命周期由 Torch 管理)
        # 格式为 (H, W, 4)，即 RGBA
        self.output_tensor = torch.zeros((self.height, self.width, 4), 
                                        dtype=torch.float32, 
                                        device=self.device)
        self.uw_output_wp = wp.from_torch(self.output_tensor)

    def process_frame(self) -> torch.Tensor:
        """
        处理当前帧，返回用于 UI 显示的 uint8 Tensor (GPU)
        """
        # 1. 强制更新传感器数据
        self.sensor.update(dt=0.01)
        
        # 2. 检查数据有效性
        if "rgb" not in self.sensor.data.output or "distance_to_camera" not in self.sensor.data.output:
            return None

        # 获取原始数据 (Environment 0)
        rgb_raw = self.sensor.data.output["rgb"][0]
        depth_raw = self.sensor.data.output["distance_to_camera"][0]

        # 3. 数据预处理 (归一化并转为 Warp 兼容格式)
        rgb = rgb_raw.float().contiguous() / 255.0
        depth = depth_raw.float().contiguous()
        if depth.dim() == 3: 
            depth = depth.squeeze(-1)

        wp_rgb = wp.from_torch(rgb, dtype=wp.float32)
        wp_depth = wp.from_torch(depth, dtype=wp.float32)

        # 4. 运行 Warp 核函数进行水下效果渲染
        wp.launch(
            kernel=UW_render_kernel,
            dim=(self.height, self.width),
            inputs=[
                wp_rgb, 
                wp_depth, 
                self.backscatter_val, 
                self.atten_coeff, 
                self.backscatter_coeff
            ],
            outputs=[self.uw_output_wp],
            device=self.device
        )

        # 5. 后处理：Clamp 并在 GPU 上转换为 uint8
        # 注意：这里直接操作 self.output_tensor，因为它是 uw_output_wp 的底层 torch 视图
        uw_uint8 = (torch.clamp(self.output_tensor, 0.0, 1.0) * 255.0).to(torch.uint8)
        
        return uw_uint8.contiguous()