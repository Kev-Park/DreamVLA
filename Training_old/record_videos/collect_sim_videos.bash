# Run this script 10 times
for i in {1..10}
do
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_eval.py --task=Isaac-Motion-Tracking-Pick-v0 --headless --video --num_envs 1000 --device cuda:1 --baseline b --id $i
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_eval.py --task=Isaac-Motion-Tracking-Punch-v0 --headless --video --num_envs 1000 --device cuda:1 --baseline b --id $i
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_eval.py --task=Isaac-Motion-Tracking-Kick-v0 --headless --video --num_envs 1000 --device cuda:1 --baseline b --id $i
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_eval.py --task=Isaac-Motion-Tracking-Button-Press-v0 --headless --video --num_envs 1000 --device cuda:1 --baseline b --id $i
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_eval.py --task=Isaac-Motion-Tracking-Jump-v0 --headless --video --num_envs 1000 --device cuda:1 --baseline b --id $i
done