cd src
# train
python main.py ctdet_angle --exp_id retail50k_dla_512 --dataset retail50k --input_res 512 --num_epochs 120 --lr_step 60,90 --gpus 0,1,2 --batch_size 128 --wh_weight 0.5 --num_workers 16 --aug_rot 0.5 --rotate 30.0

# piou
python main.py ctdet_angle --exp_id retail50k_dla_piou_512 --dataset retail50k --input_res 512 --num_epochs 60 --lr 5e-5 --lr_step 30,60 --gpus 0,1,2 --batch_size 128 --wh_weight 0.5 --num_workers 16 --aug_rot 0.5 --rotate 30.0 --piou_weight 1.0 --resume --load_model ../exp/ctdet_angle/retail50k_dla_512/model_best.pth --val_intervals 20


# --arch resdcn_101
# python main.py ctdet_angle --exp_id retail50k_resdcn_101_512 --dataset retail50k --arch resdcn_101 --input_res 512 --num_epochs 120 --lr_step 60,90 --gpus 0,1,2 --batch_size 84 --wh_weight 0.5 --num_workers 32 --aug_rot 0.5 --rotate 30.0 --val_intervals 20

# # --arch resdcn_18
# python main.py ctdet_angle --exp_id retail50k_resdcn_18_512 --dataset retail50k --arch resdcn_18 --input_res 512 --num_epochs 120 --lr_step 60,90 --gpus 0,1,2 --batch_size 128 --wh_weight 0.5 --num_workers 32 --aug_rot 0.5 --rotate 30.0 --val_intervals 20

# python3.5 main.py ctdet_angle --exp_id dota_dla_512 --dataset dota --input_res 512 --num_epochs 120 --lr_step 60,90 --gpus 0,1 --batch_size 2 --wh_weight 0.5
