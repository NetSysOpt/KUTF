cd ../src;
python train_gnn.py -t rlp_gnnk1 -m 10;
python predict_gnn.py -t rlp_gnnk1;
cd ./julia/PDLP;
python gen_bat.py -t lplarge_test -e rlp_gnn_sup_s0 -m 1800;
bash run_test.bat;
cd ../../logs;
python analyze.py -t lplarge -e rlp_gnn_sup_s0;
cd ../../script;