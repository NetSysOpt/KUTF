cd ../src;
python train_new.py -t mm_gnnk1_sum -m 500;
python predict_new.py -t mm_gnnk1_sum;
cd ./julia/PDQP.jl;
python gen_bat.py -t mm_test -o 1 -e mm_gnnk1_sum;
bash run_test.bat;
cd ../../logs;
python analyze.py -t mm -e mm_gnnk1_sum;
cd ../../script;
cd ../src;
python train_new.py -t mm_gnnk4 -m 300;
python predict_new.py -t mm_gnnk4;
cd ./julia/PDQP.jl;
python gen_bat.py -t mm_test -o 1 -e mm_gnnk4;
bash run_test.bat;
cd ../../logs;
python analyze.py -t mm -e mm_gnnk4;
cd ../../script;
cd ../src;
python train_new.py -t mm_gnnk8 -m 300;
python predict_new.py -t mm_gnnk8;
cd ./julia/PDQP.jl;
python gen_bat.py -t mm_test -o 1 -e mm_gnnk8;
bash run_test.bat;
cd ../../logs;
python analyze.py -t mm -e mm_gnnk8;
cd ../../script;


