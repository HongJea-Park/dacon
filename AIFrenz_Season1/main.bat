@echo on

python main.py --epoch 100 --lr 1e-3 --weight_decay 1e-3 --step_size 3
timeout /t -1
python main.py --epoch 100 --lr 1e-4 --weight_decay 1e-2 --fine_tune True
python predict.py

timeout /t -1