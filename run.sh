

for seed in 42 43 44 45 46; do

  python main.py \
    --save_dir exp/resnet34+oversampling+use_mixup+seed_${seed} \
    --oversampling \
    --use_mixup \
    --model resnet34 \
    --seed $seed 
    
done



