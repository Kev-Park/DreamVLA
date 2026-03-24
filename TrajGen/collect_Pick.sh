conda activate omnicontrol
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --task pick --output_dir sample/Pick_sim --batch_size 128
conda activate dreamcontrol
cd sample/Pick_sim1
rm -r *.pkl
cd ../Pick_sim2
rm -r *.pkl
cd ../Pick_sim_hum
rm -f *.pkl
cd ../Pick_sim
python3 retarget.py
cd ../Pick_sim1
python3 refine_motions_al.py
cd ..
