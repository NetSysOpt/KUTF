cd ../src;
python reshufflepkl.py -t mm -s 1;
python train_new.py -t mm1 -m 200;
python reshufflepkl.py -t mm -s 2;
python train_new.py -t mm2 -m 200;
cd ../script