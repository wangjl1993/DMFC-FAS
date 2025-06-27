

for seed in 42 43 44 45 46; do

  python main.py \
    --data_file ./prepare_data/croped_data/train.csv \
    --save_dir result/resnet34+oversampling+use_mixup+seed_${seed} \
    --oversampling \
    --use_mixup \
    --model resnet34 \
    --seed $seed \
    --remove_illegal_faces
  
  python main.py \
    --data_file ./prepare_data/croped_data/val_test.csv \
    --save_dir result/resnet34+oversampling+use_mixup+seed_${seed} \
    --mode inference \
    --model resnet34 \
    --resume result/resnet34+oversampling+use_mixup+seed_${seed}/model_epoch_150.pth
done



