conda activate omnicontrol
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --task punch_real --output_dir sample/punch_real --batch_size 1500
conda activate dreamcontrol
cd sample/Punch_real1
# rm -r *.pkl
cd ../Punch_real2
rm -r *.pkl
cd ../Punch_real
python3 retarget.py
cd ../Punch_real1
python3 refine_motions.py
cd ..
