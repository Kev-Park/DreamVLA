conda activate omnicontrol
python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt --num_repetitions 1 --task button_press_real --output_dir sample/Button_Press_real --batch_size 1500
conda activate dreamcontrol
cd sample/Button_Press_real1
# rm -r *.pkl
cd ../Button_Press_real2
rm -r *.pkl
cd ../Button_Press_real
python3 retarget.py
cd ../Button_Press_real1
python3 refine_motions.py
cd ..
