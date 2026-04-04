conda activate omnicontrol
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --task kick --output_dir sample/Kick_sim --batch_size 128
conda activate dreamcontrol_51
cd sample/Kick_sim1
rm -r *.pkl
cd ../Kick_sim2
rm -r *.pkl
cd ../Kick_sim
python3 retarget.py
cd ../Kick_sim1
python3 refine_motions.py
cd ..
