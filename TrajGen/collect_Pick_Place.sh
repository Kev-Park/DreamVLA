conda activate dreamcontrol_51
cd sample/Pick_Place_sim1
rm -r *.pkl
cd ../Pick_Place_sim2
rm -r *.pkl
cd ../Pick_Place_sim
python3 retarget.py
cd ../Pick_Place_sim1
python3 refine_motions.py
cd ..
