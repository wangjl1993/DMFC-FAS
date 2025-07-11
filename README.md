# Adaptive Face Anti-Spoofing Through Enhanced Data Manipulation and Feature Alignment Techniques

## Data Preparation
Navigate to the `prepare_data` directory and follow the instructions in the README.md file to prepare the data.

## Environment Setup
Install the training environment according to `requirements.txt`. Note that this may differ from `prepare_data/requirements.txt`, so please ensure you install the correct dependencies.

## Training and Inference

See run.sh

```bash
for seed in 42 43 44 45 46; do

  python main.py \
    --save_dir exp/resnet34+oversampling+use_mixup+seed_${seed} \
    --oversampling \
    --use_mixup \
    --model resnet34 \
    --seed $seed 
    
done
```
This will generate a `scores.txt` file in the `exp/resnet34+oversampling+use_mixup+seed_${seed}` directory, containing prediction scores for both validation and test sets.
