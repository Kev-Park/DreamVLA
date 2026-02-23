conda activate omnicontrol
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --task pick_real --output_dir sample/Pick_real --batch_size 1500
conda activate dreamcontrol
cd sample/Pick_real1
# rm -r *.pkl
cd ../Pick_real2
rm -r *.pkl
cd ../Pick_real
python3 retarget.py
cd ../Pick_real1
python3 refine_motions.py
cd ..
