conda activate dreamcontrol_51
cd sample/Jump_sim1
rm -r *.pkl
cd ../Jump_sim2
rm -r *.pkl
cd ../Jump_sim
python3 retarget.py
cd ../Jump_sim1
python3 refine_motions.py
cd ..
