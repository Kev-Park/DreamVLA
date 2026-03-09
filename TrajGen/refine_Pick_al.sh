conda activate dreamcontrol

cd sample/Pick_sim2
rm -r *.pkl
cd ../Pick_sim1
python3 refine_motions_al.py
cd ..
