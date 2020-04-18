@echo on

start visdom_server.bat

python main.py --model_name target_eda --Y_list Y00
python main.py --model_name target_eda --Y_list Y01
python main.py --model_name target_eda --Y_list Y02
python main.py --model_name target_eda --Y_list Y03
python main.py --model_name target_eda --Y_list Y04
python main.py --model_name target_eda --Y_list Y05
python main.py --model_name target_eda --Y_list Y06
python main.py --model_name target_eda --Y_list Y07
python main.py --model_name target_eda --Y_list Y08
python main.py --model_name target_eda --Y_list Y09
python main.py --model_name target_eda --Y_list Y10
python main.py --model_name target_eda --Y_list Y11
python main.py --model_name target_eda --Y_list Y12
python main.py --model_name target_eda --Y_list Y13
python main.py --model_name target_eda --Y_list Y14
python main.py --model_name target_eda --Y_list Y15
python main.py --model_name target_eda --Y_list Y16
python main.py --model_name target_eda --Y_list Y17

python main.py --transfer False --predict False --Y_list Y12
python main.py --weight_decay 0.1 --transfer True --predict True --filename submission_y12_tl_36 --Y_list Y12

python main.py --transfer False --predict False --Y_list Y15
python main.py --weight_decay 0.1 --transfer True --predict True --filename submission_y15_tl_36 --Y_list Y15

python main.py --transfer False --predict False --Y_list Y12,Y15
python main.py --lr 1e-2 --chunk_size 36 --weight_decay 0.5 --transfer True --predict True --filename submission_y12_y15_tl_36 --Y_list Y12,Y15

timeout /t -1
