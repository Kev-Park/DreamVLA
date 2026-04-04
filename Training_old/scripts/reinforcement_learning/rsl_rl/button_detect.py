import logging
import time

import cv2
import numpy as np
from grid_cortex_client.cortex_client import CortexClient

 

MODEL_ID = "owlv2"

logging.getLogger("grid_cortex_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import base64
import json
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import time
import requests 
import time



def get_box_center(box):
    """
    Calculate the center of a single bounding box given in xyxy format.

    Args:
        box (list or np.ndarray): A bounding box with format [x_min, y_min, x_max, y_max].

    Returns:
        tuple: Center point (x_center, y_center).
    """
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return x_center, y_center

def call_owl_cortex(rgb, obj):
    with CortexClient(base_url="http://34.87.140.238:8000/") as client:
        start = time.time()  # Start timing
        output = client.run(
            model_id=MODEL_ID, image_input=rgb, prompt=obj, debug=True
        )
        print(
            f"Time taken for {MODEL_ID}: {(time.time() - start) * 1000:.2f} ms"
        )  # Log the time taken
        print(f"SUCCESS: Model '{MODEL_ID}' ran successfully.")
        boxes = output["boxes"]
        scores = output["scores"]
    return boxes, scores

def encode_image(rgb_array):
    """Converts RGB array to base64 string."""
    # Convert RGB array to PIL Image
    if rgb_array.dtype != np.uint8:
        rgb_array = rgb_array.astype(np.uint8)
    
    pil_image = Image.fromarray(rgb_array)
    
    # Convert to bytes
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    
    # Encode to base64
    return base64.b64encode(image_bytes).decode("utf-8")


def call_nano_owl(rgb, obj):
    url = "http://localhost:8001/run"
    
    # Encode the RGB array directly
    image_b64 = encode_image(rgb)

    # Prepare the payload as expected by the endpoint
    payload = {
        "image_input": image_b64,
        "prompt": f"[{obj}]",
    }
    
    # Send a POST request with the JSON payload
    start = time.time()
    response = requests.post(url, json=payload)
    print(f"Time taken for nano_owl: {(time.time() - start) * 1000:.2f} ms")
    
    if response.status_code == 200:
        print(f"SUCCESS: Nano OWL ran successfully.")
        result = response.json()
        boxes = result["boxes"]
        scores = result["scores"]
        return boxes, scores
    else:
        print(f"Request failed with status code {response.status_code}")
        print("Response:", response.text)
        return None, None

def getXyzInHandFrame(mid_x, mid_y, depth_in_meters, color_intrinsics):
    x_center_ = (
        (mid_x - color_intrinsics["ppx"])
        / color_intrinsics["fx"]
        * depth_in_meters
    )
    y_center_ = (
        (mid_y - color_intrinsics["ppy"])
        / color_intrinsics["fy"]
        * depth_in_meters
    )
    z_center_ = depth_in_meters
    x_center = LOC_X + z_center_ - X_HAND
    y_center = LOC_Y - x_center_
    z_center = LOC_Z - y_center_
    print(
        f"3D coordinates of the center: ({x_center}, {y_center}, {z_center})"
    )
    if isinstance(x_center, list):
        x_center = x_center[0]
    if isinstance(y_center, list):
        y_center = y_center[0]
    if isinstance(z_center, list):
        z_center = z_center[0]


    return [x_center, y_center, z_center]


def detect_item_iteration(agent, color_intrinsics, obj):
    rgb = agent.sensors["rgb"].getImage()
    depth = agent.sensors["depth"].getImage()
    print("got rgb with type:", type(rgb))
    print("got depth with type:", type(depth))
    time.sleep(1)
    if rgb is not None and rgb.data is not None:
        rgb_data = rgb.data
        depth_data = depth.data

        # Print debug info about the images
        print(f"RGB shape: {rgb_data.shape}, depth shape: {depth_data.shape}")
        if hasattr(rgb, "capture_params"):
            print(f"RGB capture params: {rgb.capture_params}")
        if hasattr(depth, "capture_params"):
            print(f"Depth capture params: {depth.capture_params}")

        # Fix: cv2.resize expects (width, height), not (height, width)
        depth_width = depth_data.shape[1]  # width
        depth_height = depth_data.shape[0]  # height
        rgb_resized = cv2.resize(rgb_data, (depth_width, depth_height))

        print(f"Resized RGB shape: {rgb_resized.shape}")

        data = rgb_resized.copy()
        boxes, scores = call_owl_cortex(rgb_resized, obj)
        if boxes is not None and len(boxes) > 0:
            print("scores: ", scores)
            i_max = np.argmax(scores)
            x_min, y_min, x_max, y_max = boxes[i_max]
            if i_max != -1:
                mid_x, mid_y = get_box_center(boxes[i_max])
                print("mid_x, mid_y: ", mid_x, mid_y)

                # Draw a red bounding box and center dot
                rgb_with_dot = rgb_resized.copy()

                # Draw bounding box in red
                cv2.rectangle(
                    rgb_with_dot,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    (255, 0, 0),  # Red color in RGB
                    2,
                )  # Line thickness

                # Draw center dot in red
                cv2.circle(rgb_with_dot, (int(mid_x), int(mid_y)), 5, (255, 0, 0), -1)

                # Fix: Convert RGB back to BGR for cv2.imwrite (OpenCV expects BGR)
                bgr_with_dot = cv2.cvtColor(rgb_with_dot, cv2.COLOR_RGB2BGR)
                cv2.imwrite("test.png", bgr_with_dot)
                print(
                    f"Saved test.png with bounding box and center dot at ({int(mid_x)}, {int(mid_y)})"
                )

                # Calculate average depth over the bounding box area
                x_min_int, y_min_int = int(x_min), int(y_min)
                x_max_int, y_max_int = int(x_max), int(y_max)
                
                # Extract the region of interest from depth data
                depth_roi = depth_data[y_min_int:y_max_int, x_min_int:x_max_int]
                
                # Calculate average depth, excluding zero values
                valid_depths = depth_roi[depth_roi != 0]
                if len(valid_depths) > 0:
                    depth_value = np.mean(valid_depths)
                else:
                    depth_value = 0
                
                print(f"Average depth over bounding box ({x_min_int}:{x_max_int}, {y_min_int}:{y_max_int}): {depth_value:.2f} mm")

                # if depth_data[int(mid_y), int(mid_x)] != 0:
                if depth_value != 0:
                    if mid_x != -1 and mid_y != -1:
                        depth_in_meters = depth_value / 1000.0
                        x, y, z = getXyzInHandFrame(mid_x, mid_y, depth_in_meters, color_intrinsics)

                        x_left = 4*depth_data.shape[1]//10
                        depth_left = depth_data[depth_data.shape[0]//2, depth_data.shape[1]//4]
                        x_right = 6*depth_data.shape[1]//10
                        depth_right = depth_data[depth_data.shape[0]//2, 3*depth_data.shape[1]//4]
                        x_left, y_left, z_left = getXyzInHandFrame(x_left, depth_data.shape[0]//2, depth_left, color_intrinsics)
                        x_right, y_right, z_right = getXyzInHandFrame(x_right, depth_data.shape[0]//2, depth_right, color_intrinsics)
                        yaw = (x_left - x_right) / (y_left - y_right)
                        print("yaw: ", yaw)
                        return [x, y, z, yaw]
                else:
                    print("Depth value is not available at the specified location")
                    bgr_no_depth = cv2.cvtColor(rgb_with_dot, cv2.COLOR_RGB2BGR)
                    cv2.imwrite("test.png", bgr_no_depth)
                    print("Saved test.png (no valid depth at the drawn point)")
                    return None

            else:
                print("No valid box detected")
                # Save image without detection
                bgr_no_detection = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)
                cv2.imwrite("test.png", bgr_no_detection)
                print("Saved test.png (no valid detection)")
                return None
        else:
            print("No boxes detected")
            # Save image without detection
            bgr_no_detection = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)
            cv2.imwrite("test.png", bgr_no_detection)
            print("Saved test.png (no boxes found)")
            return None
    else:
        print("invalid image")
        return None


def detect_item_loop(agent, color_intrinsics, obj):
    while True:
        item_3d_location = detect_item_iteration(agent, color_intrinsics, obj)
        if item_3d_location is not None:
            print("Item 3D location: ", item_3d_location)
            return item_3d_location
        else:
            print("No item detected")
            time.sleep(0.1)


def main():
    agent = G1("eth0")
    rgb_cam = ZenohCamera("color/image_raw", auto_parse_ros2=True)
    depth_cam = ZenohCamera("depth/image_rect_raw", auto_parse_ros2=True)
    agent.addSensor("rgb", rgb_cam)
    agent.addSensor("depth", depth_cam)
    intrinsics = {
        "fx": 606.00341796875,
        "fy": 605.660888671875,
        "ppx": 128,
        "ppy": 128,
    }
    print(detect_item_loop(agent, intrinsics, "elevator service button"))


if __name__ == "__main__":
    main()
