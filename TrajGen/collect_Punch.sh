conda activate omnicontrol
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --task punch --output_dir sample/Punch_sim --batch_size 128
conda activate dreamcontrol_51
cd sample/Punch_sim1
rm -r *.pkl
cd ../Punch_sim2
rm -r *.pkl
cd ../Punch_sim
python3 retarget.py
cd ../Punch_sim1
python3 refine_motions.py
cd ..
