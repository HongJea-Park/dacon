@echo on

python main.py --epoch 200 --weight_decay 1e-2 --n_layers 2 --n_hidden 10 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 2 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 2 --n_hidden 10 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 2 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-2 --n_layers 2 --n_hidden 20 --lr 1e-3 
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 2 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 2 --n_hidden 20 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 2 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-2 --n_layers 3 --n_hidden 10 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 3 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 3 --n_hidden 10 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 3 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-2 --n_layers 3 --n_hidden 20 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 3 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 3 --n_hidden 20 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 3 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-2 --n_layers 4 --n_hidden 10 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 4 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 4 --n_hidden 10 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 4 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-2 --n_layers 4 --n_hidden 20 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 4 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 4 --n_hidden 20 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-2 --n_layers 4 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False --c_loss False


python main.py --epoch 200 --weight_decay 1e-3 --n_layers 2 --n_hidden 10 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 2 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 2 --n_hidden 10 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 2 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-3 --n_layers 2 --n_hidden 20 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 2 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 2 --n_hidden 20 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 2 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-3 --n_layers 3 --n_hidden 10 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 3 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 3 --n_hidden 10 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 3 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-3 --n_layers 3 --n_hidden 20 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 3 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 3 --n_hidden 20 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 3 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-3 --n_layers 4 --n_hidden 10 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 4 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 4 --n_hidden 10 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 4 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-3 --n_layers 4 --n_hidden 20 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 4 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 4 --n_hidden 20 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-3 --n_layers 4 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False --c_loss False


python main.py --epoch 200 --weight_decay 1e-4 --n_layers 2 --n_hidden 10 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 2 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 2 --n_hidden 10 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 2 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-4 --n_layers 2 --n_hidden 20 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 2 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 2 --n_hidden 20 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 2 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-4 --n_layers 3 --n_hidden 10 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 3 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 3 --n_hidden 10 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 3 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-4 --n_layers 3 --n_hidden 20 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 3 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 3 --n_hidden 20 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 3 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-4 --n_layers 4 --n_hidden 10 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 4 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 4 --n_hidden 10 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 4 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-4 --n_layers 4 --n_hidden 20 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 4 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 4 --n_hidden 20 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-4 --n_layers 4 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False --c_loss False


python main.py --epoch 200 --weight_decay 1e-5 --n_layers 2 --n_hidden 10 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 2 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 2 --n_hidden 10 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 2 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-5 --n_layers 2 --n_hidden 20 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 2 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 2 --n_hidden 20 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 2 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-5 --n_layers 3 --n_hidden 10 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 3 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 3 --n_hidden 10 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 3 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-5 --n_layers 3 --n_hidden 20 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 3 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 3 --n_hidden 20 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 3 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-5 --n_layers 4 --n_hidden 10 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 4 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 4 --n_hidden 10 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 4 --n_hidden 10 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

python main.py --epoch 200 --weight_decay 1e-5 --n_layers 4 --n_hidden 20 --lr 1e-3
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 4 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 4 --n_hidden 20 --lr 1e-3 --c_loss False
python main.py --epoch 200 --weight_decay 1e-5 --n_layers 4 --n_hidden 20 --lr 1e-4 --fine_tune True --early_stop False --c_loss False

timeout /t -1
