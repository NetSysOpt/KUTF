cd ../src;
python reshufflepkl.py -s 0 -t mm;
python train_new.py -t mm_gnnk1_sum -m 300;
python predict_new.py -t mm_gnnk1_sum;
cd ./julia/PDQP.jl;
python gen_bat.py -t mm_test -o 1 -e mm_gnnk1_s0;
bash run_test.bat;
cd ../../logs;
python analyze.py -t mm -e mm_gnnk1_s0;
cd ../../script;
cd ../src;
python reshufflepkl.py -s 1 -t mm;
python train_new.py -t mm_gnnk1_sum -m 300;
python predict_new.py -t mm_gnnk1_sum;
cd ./julia/PDQP.jl;
python gen_bat.py -t mm_test -o 1 -e mm_gnnk1_s1;
bash run_test.bat;
cd ../../logs;
python analyze.py -t mm -e mm_gnnk1_s1;
cd ../../script;
cd ../src;
python reshufflepkl.py -s 2 -t mm;
python train_new.py -t mm_gnnk1_sum -m 300;
python predict_new.py -t mm_gnnk1_sum;
cd ./julia/PDQP.jl;
python gen_bat.py -t mm_test -o 1 -e mm_gnnk1_s2;
bash run_test.bat;
cd ../../logs;
python analyze.py -t mm -e mm_gnnk1_s2;
cd ../../script;

