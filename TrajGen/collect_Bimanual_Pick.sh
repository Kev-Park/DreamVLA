conda activate dreamcontrol_51
cd sample/Bimanual_Pick_sim1
rm -r *.pkl
cd ../Bimanual_Pick_sim2
rm -r *.pkl
cd ../Bimanual_Pick_sim
python3 retarget.py
cd ../Bimanual_Pick_sim1
python3 refine_motions.py
cd ..
