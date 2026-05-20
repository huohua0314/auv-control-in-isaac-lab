# 🌉 UE5 to IsaacLab Conversion

To leverage NVIDIA's GPU-accelerated reinforcement learning and advanced physics simulation, we migarate oceangym from UE5 to IsaacLab.

You can find the related conversion and control scripts in the `OceanGym2isaac` directory.

### Step 1: Set up ernvironment
To ensure the simulation runs correctly with GPU acceleration, you must first install **NVIDIA Isaac Sim** and **Isaac Lab**.

1.  **System Requirements:** A Linux-based system (Ubuntu 20.04/22.04) with an NVIDIA GPU (RTX 30-series or higher recommended).
2.  **Follow the Official Installation Guide:**  
    Please complete the setup by following the step-by-step instructions here:  
    👉 **[Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)**

> [!TIP]
> We recommend using the **Conda** installation method provided in the guide to ensure all dependencies (like PyTorch and NVIDIA Warp) are correctly managed.

### Step 2: Download Oceangym_Isaac.zip
From 
  ☁️ <a href="https://drive.google.com/file/d/1EfKHeiyQD5eoJ6-EsiJHuIdBRM5Ope5A/view?usp=drive_link" target="_blank">Google Drive</a>
  ☁️ <a href="https://pan.baidu.com/s/16h86huHLeFGAKatRWvLrFQ?pwd=wput" target="_blank">Baidu Drive</a>
  download the **OceanGym_Isaac.zip** And extract it to the folder you want

### Step 3: Quick Start

Navigate to the `OceanGym2isaac` directory. You can test the simulation using either keyboard control or the VLA model.

> [!IMPORTANT]
> **Path Configuration:** Before running the simulation, you must modify `auv_config.yaml` to set the **absolute paths** for the simulation assets. 
> - Update `auv_usd` to the absolute path of **`auv.usd`** (located in the `mesh/` folder).
> - Update `floor_usd` to the absolute path of **`floor_collision.usd`** (located in the `final_floor/` folder).
>
> *Example:* `/home/user/OceanGym2isaac/asset/mesh/auv.usd`
> 
#### ⌨️ Manual Control (Keyboard)
To manually control the AUV and check the camera views, run:
```bash
python run.py --mode keyboard --enable_cameras
```
Keyboard Control Mapping:
| Key | function |
| ---| --- |
|Up Arrow	|Move the AUV forward (Linear X+)
|Down Arrow	|Move the AUV backward (Linear X-)
|Left Arrow	|Turn the AUV left (Yaw+)
|Right Arrow	T|urn the AUV right (Yaw-)
|Page Up	|Ascend the AUV (Vertical Z+)
|Page Down	|Descend the AUV (Vertical Z-)

#### 🤖 Autonomous Task (VLA Model)
To call a Large Model (Vision-Language-Action) to perform a simple autonomous docking/entry task:
```bash
python run.py --mode vla --enable_cameras
```
> [!NOTE]
> You need to modify the YAML file first to set the required runtime parameters.




### 🏋️ Reinforcement Learning (RL) Workflow

Follow these steps to configure, train, and evaluate the ROV using Proximal Policy Optimization (PPO) via the `skrl` framework.

#### 1. Navigate to the Task Directory
First, ensure you are inside the reinforcement learning environment directory:
```bash
cd OceanGym2isaac/isaac_rl

```

#### 2. Download and Place Assets (USD)

1. Download the **asset.zip** package from ☁️ Baidu Drive.
2. Extract the contents. Ensure the `usd/` folder and its underlying files (`BlueROV1.usd`, `BlueROV.usd`, etc.) are placed under your root or corresponding project path as shown in the project structure.

#### 3. Path Configuration (`config.yaml`)

Before starting any training or evaluation, you must update the asset paths in `config.yaml` to match your local absolute paths. Open `config.yaml` and modify the following fields:

```yaml
paths:
  sea_floor_usd: "absolute path of floor_collison.usd" (located in the `final_floor/` folder)
  robot_usd: "absolute path of BlueROV1.usd" (located in the `usd/` folder)

```

#### 4. Policy Training

To launch parallel training across multiple environments using PPO:

```bash
python train.py --task Isaac-ROV-Docking-Direct-v0 --num_envs 64 --headless

```

* **Visualization:** Remove the `--headless` flag if you want to bring up the Isaac Sim GUI and watch the ROV learn in real-time.
* **Monitoring:** Training logs, tensorboard telemetry, and model checkpoints will be stored under the `runs/` directory. Monitor the training progress by running:
```bash
tensorboard --logdir runs

```

#### 5. Policy Evaluation & Testing
To test and visualize your trained model checkpoint (e.g., the saved `best_agent.pt` file):
```bash
python evaluate.py --task Isaac-ROV-Docking-Direct-v0 --checkpoint runs/ROV_Docking_Train/checkpoints/best_agent.pt --num_envs 1

```

> [!TIP]
> The evaluation script automatically switches the agent to test mode (`agent.set_running_mode("test")`). This turns off stochastic Gaussian exploration, forcing the ROV to execute purely deterministic, optimized actions based on your learned weights.




```

```
