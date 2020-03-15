@echo on

python main.py --epoch 300 --lr 1e-3 --early_stop True
python main.py --epoch 300 --lr 1e-4 --fine_tune True

timeout /t -1