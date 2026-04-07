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

target_item_image_path = "/home/huohua/IsaacLab/ocean/project/test1.png"

base64_image_data = image_path_to_base64(target_item_image_path)
variables = {
    "target_item": "Landing platforms with orange border",
    "target_item_image": base64_image_data,
    "target_item_description": "A landing platform with oranger border used for robot charging,",
    "task_description": "Find the landing platform with oranger border and LAND SAFELY on it. The primary goal is precise landing - continuously adjust your position to keep the platform EXACTLY in the CENTER of your down camera and then keep descending. Make small, careful movements to maintain this center alignment throughout the approach. After successful landing, report the exact location to the user. IMPORTANT: The mission is only considered complete after successful landing with the platform kept centered in your camera view, not just finding the platform. After landing, the robot should stop all movement to complete the docking procedure.",
    "another_target_photo": None
}

# Format the prompt
formatted_prompt = prompt_template.format(**variables)