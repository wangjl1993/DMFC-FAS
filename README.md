# Adaptive Face Anti-Spoofing Through Enhanced Data Manipulation and Feature Alignment Techniques

## Data Preparation
Navigate to the `prepare_data` directory and follow the instructions in the README.md file to prepare the data.

## Environment Setup
Install the training environment according to `requirements.txt`. Note that this may differ from `prepare_data/requirements.txt`, so please ensure you install the correct dependencies.

## Training and Inference

See run.sh

```bash
for seed in 42 43; do

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
```
This will generate a `scores.txt` file in the `result/resnet34+oversampling+use_mixup+seed_${seed}/model_epoch_150_extracted_features_val_test` directory, containing prediction scores for both validation and test sets.
