import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import argparse

label_map = {
    '0_0_0': ('Live Face', 'Live'),
    '1_0_0': ('Print', '2D Attack'),
    '1_0_1': ('Replay', '2D Attack'),
    '1_0_2': ('Cutouts', '2D Attack'),
    '1_1_0': ('Transparent', '3D Attack'),
    '1_1_1': ('Plaster', '3D Attack'),
    '1_1_2': ('Resin', '3D Attack'),
    '2_0_0': ('Attibute-Edit', 'Digital Manipulation'),
    '2_0_1': ('Face-Swap', 'Digital Manipulation'),
    '2_0_2': ('Video-Driven', 'Digital Manipulation'),
    '2_1_0': ('Pixcel-Level', 'Digital Adversarial'),
    '2_1_1': ('Semantic-Leve', 'Digital Adversarial'),
    '2_2_0': ('ID_Consisnt', 'Digital Generation'),
    '2_2_1': ('Style', 'Digital Generation'),
    '2_2_2': ('Prompt', 'Digital Generation'),
}


def generate_csv_file(protocol_file, output_csv, label_map):
    rows = []
    with open(protocol_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f'Processing {protocol_file}'):
            line = line.strip()
            if not line:
                continue
            
            # for train/val data
            if len(line.split()) == 2:
                filename, code = line.split()
            
            # for test data
            else:
                filename = line.split()[0]
                code = 'Unknown'

            sublabel, label = label_map.get(code, ('Unknown', 'Unknown'))
            image_path = filename
            try:
                with Image.open(image_path) as img:
                    size = f"{img.width}x{img.height}"
            except Exception:
                size = "NA"
            rows.append([filename, sublabel, label, size])
    df = pd.DataFrame(rows, columns=['filename', 'sublabel', 'classlabel', 'image_size'])
    df.to_csv(output_csv, index=False, encoding='utf-8')
    # 输出sublabel分布
    print(f"Processing {protocol_file} completed. Output saved to {output_csv}.")
    print(f"Sublabel distribution:\n{df['sublabel'].value_counts()}\n")




def parse_args():
    parser = argparse.ArgumentParser(description='prepare data_file')
    parser.add_argument('--data_dir', type=str, default='/home/akuvox2025/HDD16T/jielong.wang/fas2025_croped_face/prepare_data/raw_data', help='Path to data directory')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = Path(args.data_dir)
    
    generate_csv_file(
        protocol_file= data_dir / 'Protocol-train.txt',
        output_csv= data_dir / 'train.csv',
        label_map=label_map
    )

    generate_csv_file(
        protocol_file=data_dir / 'Protocol-val.txt',
        output_csv= data_dir / 'val.csv',
        label_map=label_map
    )

    generate_csv_file(
        protocol_file=data_dir / 'Protocol-test.txt',
        output_csv= data_dir / 'test.csv',
        label_map=label_map
    )