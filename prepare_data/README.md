# Data Preparation

## Data Directory
The data directory structure is organized as follows:
```bash
├── raw_data
│   ├── Data-test
│   ├── Data-train
│   ├── Data-val
│   ├── get_croped_data.py
│   ├── get_csv_file.py
│   ├── Protocol-test.txt
│   ├── Protocol-train.txt
│   ├── Protocol-val.txt
```

## Environment Installation
```bash
conda create -n insightface python=3.12
conda activate insightface
pip install -r requirements.txt
```

## Generate CSV Files
```bash
cd raw_data
python get_csv_file.py
```
This will generate `train.csv`, `val.csv`, and `test.csv` files in the `raw_data` directory.

## Crop and Align Faces
```bash
cd raw_data
python get_croped_data.py
```
This will create a `croped_data` folder in the `prepare_data` directory. The folder will contain `Data-test`, `Data-train`, `Data-val`, `train.csv`, and `val_test.csv` files.
