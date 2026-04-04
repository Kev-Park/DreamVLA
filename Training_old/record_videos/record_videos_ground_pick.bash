#!/bin/bash

PARENT_FOLDER="Videos"
# TASK_FOLDER="pick_videos_new"
TASK_FOLDER="ground_pick_videos"
# TASK_FOLDER="bimanual_pick_videos"
# TASK_FOLDER="ground_pick_new_videos"
# TASK_FOLDER="punch_videos"
# TASK_FOLDER="kick_videos"
# TASK_FOLDER="jump_videos"
# TASK_FOLDER="button_press_videos1"
# TASK_FOLDER="pick_place_videos"
# TASK_FOLDER="sit_videos"
# TASK_FOLDER="open_drawer_videos"

# List of object names
# OBJECT_NAMES=("mustard_bottle" "sugar_box1" "mug" "tomato_soup" "cheeze_it" "can_meat" "pitcher")
# OBJECT_NAMES=("pitcher_big")
# OBJECT_NAMES=("package")
# OBJECT_NAMES=("cheeze_it_ground")
# OBJECT_NAMES=("none")
# OBJECT_NAMES=("none")
# OBJECT_NAMES=("none")
OBJECT_NAMES=("none")
# OBJECT_NAMES=("pitcher")
# OBJECT_NAMES=("none")
# OBJECT_NAMES=("none")

# CMD="./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_camera.py --task=Isaac-Motion-Tracking-Pick-Play-v0  --headless --num_envs 1 --device cuda:1 --enable_cameras --path "logs/rsl_rl/g1_flat/2025-09-11_20-20-45-Pick-a/model_4550.pt" --rendering_mode quality"
CMD="./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_camera.py --task=Isaac-Motion-Tracking-Ground-Pick-Play-v0  --headless --num_envs 1 --device cuda:0 --enable_cameras --rendering_mode quality --path "/home/dvij/isaaclab-sparky/logs/rsl_rl/g1_flat/2025-07-30_17-22-09-Ground-Pick-a/model_1950.pt""
# CMD="./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_camera.py --task=Isaac-Motion-Tracking-Bimanual-Pick-Play-v0  --headless --num_envs 1 --device cuda:0 --enable_cameras --path "/home/dvij/isaaclab-sparky/logs/rsl_rl/g1_flat/2025-07-30_11-52-34-Bimanual-Pick-a/model_1900.pt" --rendering_mode quality"
# CMD="./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_camera.py --task=Isaac-Motion-Tracking-Ground-Pick-New-Play-v0  --headless --num_envs 1 --device cuda:0 --enable_cameras"
# CMD="./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_camera.py --task=Isaac-Motion-Tracking-Punch-Play-v0  --headless --num_envs 1 --device cuda:1 --enable_cameras --path /home/dvij/isaaclab-sparky/logs/rsl_rl/g1_flat/2025-07-31_13-39-04/model_850.pt"
# CMD="./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_camera.py --task=Isaac-Motion-Tracking-Kick-Play-v0  --headless --num_envs 1 --device cuda:1 --enable_cameras"
# CMD="./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_camera.py --task=Isaac-Motion-Tracking-Jump-Play-v0  --headless --num_envs 1 --device cuda:1 --enable_cameras --path /home/dvij/isaaclab-sparky/logs/rsl_rl/g1_flat/2025-07-31_14-33-17/model_jump.pt"
# CMD="./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_camera.py --task=Isaac-Motion-Tracking-Button-Press-Play-v0  --headless --num_envs 1 --device cuda:1 --enable_cameras --path /home/dvij/isaaclab-sparky/logs/rsl_rl/g1_flat/2025-08-12_21-03-36-Button-Press/model_1499.pt"
# CMD="./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_camera.py --task=Isaac-Motion-Tracking-Pick-Place-Play-v0  --headless --num_envs 1 --device cuda:1 --enable_cameras --video_length 550"
# CMD="./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_camera.py --task=Isaac-Motion-Tracking-Sit-Play-v0  --headless --num_envs 1 --device cuda:1 --enable_cameras "
# CMD="./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_camera.py --task=Isaac-Motion-Tracking-Open-Drawer-Play-v0  --headless --num_envs 1 --device cuda:1 --enable_cameras "

# Loop through each object
for OBJECT_NAME in "${OBJECT_NAMES[@]}"
do
    # Path to your Python script
    FOLDER_NAME="$PARENT_FOLDER/$TASK_FOLDER/$OBJECT_NAME"

    mkdir -p "$FOLDER_NAME"
    # Loop to run the script 10 times
    for i in {1..3}
    do
        echo "Run #$i"
        FILE_NAME="$FOLDER_NAME/$i.mp4"
        # Run the command and add option --name pick_{i} to the command
        CMD_WITH_NAME="$CMD --name $FILE_NAME --object_name $OBJECT_NAME"
        echo "Executing: $CMD_WITH_NAME"
        # Execute
        $CMD_WITH_NAME
    done
done