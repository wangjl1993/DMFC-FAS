export http_proxy="http://192.168.13.168:7890"
export https_proxy="http://192.168.13.168:7890"


python main.py \
  --data_file /home/akuvox2025/SSD4T/jielong.wang/dataset/cv2025/face_phase1/train.csv \
  --save_dir exp/efficientnetb4 \
  --model efficientnetb4


python main.py \
  --data_file /home/akuvox2025/SSD4T/jielong.wang/dataset/cv2025/face_phase1/train.csv \
  --save_dir exp/efficientnetb4+focal \
  --loss focal \
  --model efficientnetb4



python main.py \
  --data_file /home/akuvox2025/SSD4T/jielong.wang/dataset/cv2025/face_phase1/train.csv \
  --save_dir exp/efficientnetb4+oversampling+use_mixup \
  --oversampling \
  --use_mixup \
  --model efficientnetb4



python main.py \
  --data_file /home/akuvox2025/SSD4T/jielong.wang/dataset/cv2025/face_phase1/train.csv \
  --save_dir exp/efficientnetb4+focal+oversampling+use_mixup \
  --loss focal \
  --oversampling \
  --use_mixup \
  --model efficientnetb4



