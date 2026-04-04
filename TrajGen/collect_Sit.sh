conda activate dreamcontrol_51
cd sample/Sit_sim1
rm -r *.pkl
cd ../Sit_sim2
rm -r *.pkl
cd ../Sit_sim
python3 retarget.py
cd ../Sit_sim1
python3 refine_motions.py
cd ..
