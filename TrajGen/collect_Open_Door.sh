conda activate dreamcontrol_51
cd sample/Open_Door_sim2
rm -r *.pkl
cd ../Open_Door_sim
python3 generate_motions_stand.py
python3 generate_motions_squat.py
cd ..
