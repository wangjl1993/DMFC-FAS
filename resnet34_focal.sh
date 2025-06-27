export http_proxy="http://192.168.13.168:7890"
export https_proxy="http://192.168.13.168:7890"



for seed in 43 44 45 46; do

  python main.py \
    --data_file /home/akuvox2025/SSD4T/jielong.wang/dataset/cv2025/face_phase1/train.csv \
    --save_dir exp/resnet34+focal+oversampling+use_mixup+seed_${seed} \
    --loss focal \
    --oversampling \
    --use_mixup \
    --model resnet34 \
    --seed $seed

done




