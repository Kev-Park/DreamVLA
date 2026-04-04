conda activate dreamcontrol_51

cd sample/Pick_sim2
rm -r *.pkl
cd ../Pick_sim1
python3 refine_motions.py
cd ..
