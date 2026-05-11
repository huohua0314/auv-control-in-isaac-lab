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
