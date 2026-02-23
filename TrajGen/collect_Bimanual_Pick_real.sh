conda activate dreamcontrol
cd sample/Bimanual_Pick_real
rm -r *.pkl
cd ../Bimanual_Pick_sim2
python3 refine_motions_for_real.py
cd ..
