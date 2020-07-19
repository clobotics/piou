cd src
# train
python main.py ctdet --exp_id coco_dla_1x --batch_size 32 --master_batch 15 --lr 1.25e-4  --gpus 0,1 --num_workers 16

# test
python test.py ctdet --exp_id coco_dla_1x --keep_res --resume
# flip test
python test.py ctdet --exp_id coco_dla_1x --keep_res --resume --flip_test 
# multi scale test
python test.py ctdet --exp_id coco_dla_1x --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
