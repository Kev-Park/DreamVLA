conda activate omnicontrol
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --task button_press --output_dir sample/button_press_sim --batch_size 128
conda activate dreamcontrol_51
cd sample/button_press_sim1
rm -r *.pkl
cd ../button_press_sim2
rm -r *.pkl
cd ../button_press_sim
python3 retarget.py
cd ../button_press_sim1
python3 refine_motions.py
cd ..
