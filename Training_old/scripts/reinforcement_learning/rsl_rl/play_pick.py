# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
import cv2
import cli_args  # isort: skip
from grid_cortex_client.cortex_client import CortexClient
import logging
import numpy as np
 
MODEL_ID = "owlv2"

logging.getLogger("grid_cortex_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--name", type=str, default="test.mp4", help="Name of the video file.")
parser.add_argument("--object_name", type=str, default=None, help="Name of the object.")
parser.add_argument("--path", type=str, default=None, help="Path to the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# exit(0)

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

# from omni.isaac.core.utils.prims import set_prim_pose
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from PIL import Image
from io import BytesIO
import base64
import requests

LOC_X = 0.05
LOC_Y = -0.04
LOC_Z = 0.36

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
    with CortexClient() as client:
        start = time.time()  # Start timing
        print(f"Calling OWL Cortex with model ID: {MODEL_ID}")
        output = client.run(
            model_id=MODEL_ID, task="detect", image_input=rgb, prompt=obj, debug=True
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

def getXyzInHandFrame(mid_x, mid_y, depth_in_meters, intrinsics):
    x_center_ = (
        (mid_x - intrinsics["ppx"])
        / intrinsics["fx"]
        * depth_in_meters
    )
    y_center_ = (
        (mid_y - intrinsics["ppy"])
        / intrinsics["fy"]
        * depth_in_meters
    )
    z_center_ = depth_in_meters
    x_center = LOC_X + z_center_
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


OUTPUT_VIDEO = "pick"

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    # exit(0)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        print(resume_path)
        import pdb; pdb.set_trace()  # noqa: E702
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        if args_cli.path:
            # if path is provided, use it to get the checkpoint
            resume_path = args_cli.path
        else :
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    
    log_dir = os.path.dirname(resume_path)
    env_cfg.viewer.eye = (1.,-2.,2.)
    env_cfg.viewer.lookat = (2., 0., 0.)
    if args_cli.object_name is not None and args_cli.object_name != "none":
        env_cfg.scene.object.spawn.usd_path = "assets/"+args_cli.object_name+".usd"

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    print(ISAACLAB_NUCLEUS_DIR)
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # camera = Camera(
    #     CameraCfg(prim_path="/world/Camera")
    # )
    # camera.initialize()
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt
    print("dt:", dt)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    from isaaclab.sensors import Camera
    
    import numpy as np
    # import Camera
    cam = env.unwrapped.scene["camera"]
    cam_robot = env.unwrapped.scene["camera_robot"]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # MP4 codec
    video_writers = []
    for i in range(1):
        video_writer = cv2.VideoWriter(args_cli.name, fourcc, 50, (2560, 1920))
        video_writers.append(video_writer)

    video_writers_robot = []
    for i in range(1): # TO-DO look
        video_writer = cv2.VideoWriter(args_cli.name[:-4] + "_robot" + ".mp4", fourcc, 50, (512, 384))
        video_writers_robot.append(video_writer)

    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode

        # import pdb; pdb.set_trace()  # noqa: E702
        image = cam.data.output["rgb"]
        image_robot = cam_robot.data.output["rgb"] # robot face camera color data
        image_robot_depth = cam_robot.data.output["depth"] # robot face camera output
        prompt = "yellow bottle"
        if timestep < 1:
            intrinsics = {
                "fx": 512*9.1/20.955,
                "fy": 512*9.1/20.955,
                "ppx": 256,
                "ppy": 192,
            }
            rgb = image_robot[0].cpu().numpy()[:,:,:]
            depth = image_robot_depth[0].cpu().numpy()[:,:,:]
            print("Before calling OWL Cortex")
            rgb_resized = cv2.resize(rgb, (512, 384))
            boxes, scores = call_owl_cortex(rgb_resized, prompt)
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
                    depth_ = (255*depth[:,:,0])
                    depth_ = np.clip(depth_, 0, 255)
                    cv2.imwrite("test_depth.png", depth_.astype(np.uint8))
                    print(
                        f"Saved test.png with bounding box and center dot at ({int(mid_x)}, {int(mid_y)})"
                    )

                    # Calculate average depth over the bounding box area
                    x_min_int, y_min_int = int(x_min), int(y_min)
                    x_max_int, y_max_int = int(x_max), int(y_max)
                    
                    # Extract the region of interest from depth data
                    depth_roi = depth[int(mid_y-5):int(mid_y+5), int(mid_x-5):int(mid_x+5)]

                    # Calculate average depth, excluding zero values
                    valid_depths = depth_roi[depth_roi != 0]
                    if len(valid_depths) > 0:
                        depth_value = np.mean(valid_depths)
                    else:
                        depth_value = 0
                    
                    print(f"Average depth over bounding box ({x_min_int}:{x_max_int}, {y_min_int}:{y_max_int}): {depth_value:.2f} mm")

                    if depth_value != 0:
                        if mid_x != -1 and mid_y != -1:
                            depth_in_meters = depth_value
                            x_new, y_new, z_new = getXyzInHandFrame(mid_x, mid_y, depth_in_meters, intrinsics)
                            print(f"3D coordinates of the center: ({x_new}, {y_new}, {z_new})")
        
        
        for i in range(1):
            video_writers[i].write(image[i].cpu().numpy()[:,:,::-1])
        for i in range(1):
            video_writers_robot[i].write(image_robot[i].cpu().numpy()[:,:,::-1])
        
        # image = camera.get_image("rgb")
        with torch.inference_mode():
            obs[:,52] = x_new
            obs[:,53] = y_new
            actions = policy(obs).clone()
            obs, _, dones, _ = env.step(actions)
        if timestep == args_cli.video_length:
            break
        timestep += 1
        print(timestep)
            

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    
    # release video writers
    for video_writer in video_writers:
        # release the video writer
        print(f"[INFO] Releasing video writer: {video_writer}")
        video_writer.release()
    for video_writer in video_writers_robot:
        # release the video writer
        print(f"[INFO] Releasing video writer: {video_writer}")
        video_writer.release()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
