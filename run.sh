


# python main.py \
#   --data_file /home/akuvox2025/SSD4T/jielong.wang/dataset/cv2025/face_phase1/train.csv \
#   --save_dir exp/baseline


# python main.py \
#   --data_file /home/akuvox2025/SSD4T/jielong.wang/dataset/cv2025/face_phase1/train.csv \
#   --save_dir exp/baseline+focal \
#   --loss focal



python main.py \
  --data_file /home/akuvox2025/SSD4T/jielong.wang/dataset/cv2025/face_phase1/train.csv \
  --save_dir exp/baseline+oversampling+use_mixup \
  --oversampling \
  --use_mixup



python main.py \
  --data_file /home/akuvox2025/SSD4T/jielong.wang/dataset/cv2025/face_phase1/train.csv \
  --save_dir exp/baseline+focal+oversampling+use_mixup \
  --loss focal \
  --oversampling \
  --use_mixup


