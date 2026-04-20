import sys
import os
import cv2
import base64

from utils.prompt.format import prompt_template

def image_to_base64_128(img):
    """
    Convert an image to Base64 encoding
    """
    img_small = cv2.resize(img, (128, 128))  # Resize to 128x128 for example
    _, buffer = cv2.imencode('.png', img_small)
    return base64.b64encode(buffer).decode()

def image_path_to_base64(image_path):
    """
    Convert an image file path to Base64 encoding
    :param image_path: Path to the image file
    :return: Base64 encoded string
    """
    # Check if path exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image, check if the path is correct or if the file is corrupted: {image_path}")
    
    # Call image_to_base64 function for encoding
    return image_to_base64_128(img)

target_item_image_path = "/home/huohua/IsaacLab/ocean/project/test4.png"
added_picture =    "/home/huohua/IsaacLab/ocean/project/cam_down.jpg"
base64_image_data = image_path_to_base64(target_item_image_path)
base64_image_data_1 = image_path_to_base64(added_picture)
variables = {
    "target_item": "A grey underwater garage",
    "target_item_image": base64_image_data,
    "target_item_description": "A grey underwater garage designed for AUV docking. The garage has a distinct rectangular entrance for the AUV to enter",
    "task_description": """Locate the grey garage in your front camera and perform a precise docking maneuver.
CRITICAL VISUAL MAPPING FOR FRONT CAMERA:

If the garage entrance is in the TOP half of the front camera image -> Execute ascend.

If the garage entrance is in the BOTTOM half of the front camera image -> Execute descend.

If the garage entrance is in the LEFT half of the front camera image -> Execute move left.

If the garage entrance is in the RIGHT half of the front camera image -> Execute move right.

ONLY when the garage entrance is EXACTLY in the CENTER  of the front camera view -> Execute move forward to enter. You must ensure the  garage entrance in the EXACTLY CENTER of front camera before move forward!

Step 1: Vertical Alignment. Use one of ['descned','ascend', 'move left', 'move right'] to center the garage entrance in the front camera, please FOLLOW CRITICAL VISUAL MAPPING FOR FRONT CAMERA.
Step 2: Horizontal Insertion. Once and ONLY once the entrance is centered, use move forward to enter the garage.
Step 3: Completion. If the down, up, left , right camera is full of gery wall, meaning auv fully inside the garage. USE 'STOP' once fully inside.
Step 4: Not Found. If no camera foud garage, the auv need Stop right now. Issue 'Stop' when can;t found garage""",
    "another_target_photo": None,
    "added_item": "the picture of garage entrance exactly in the center of front camera",
    "added_picture": base64_image_data_1
}

# Format the prompt
formatted_prompt = prompt_template.format(**variables)