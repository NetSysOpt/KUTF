cd ../src;
python predict_new.py -t netlib_rlp;
cd ./julia/PDLP;
python gen_bat.py -t netlib_test -e netlib_gnn_sup -m 400 -o 1;
bash run_test.bat;
cd ../../logs;
python analyze.py -t netlib -e netlib_gnn_sup;
cd ../../script;