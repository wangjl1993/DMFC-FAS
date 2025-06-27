import os
import cv2
import pandas as pd
import numpy as np
import insightface
from insightface.utils import face_align
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def align_and_save(img, faces, save_path, size=336):
    if len(faces) > 0:
        face = faces[0]
        # 使用 insightface 的 norm_crop 进行对齐
        aligned_face = face_align.norm_crop(img, face.kps, image_size=size)
    else:
        aligned_face = cv2.resize(img, (size, size))
    cv2.imwrite(save_path, aligned_face)

def resize_and_pad(img, target_size=(640, 640), fill=0):
    img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    target_h, target_w = target_size
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(fill, fill, fill))
    return img

def main(csv_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    new_rows = []
    for idx, row in df.iterrows():
        img_path = row['filename']
        if 'downsample' in img_path or 'MotionBlur' in img_path or 'UpDownResize' in img_path:
            print(f"跳过降采样图片: {img_path}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = resize_and_pad(img, target_size=(640, 640), fill=0)
        faces = app.get(img)
        save_path = os.path.join(save_dir, img_path)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        align_and_save(img, faces, save_path)
        print(f"保存: {save_path}, faces: {len(faces)}")
        new_rows.append({
            'filename': row['filename'],
            'sublabel': row['sublabel'],
            'classlabel': row['classlabel'],
            'image_size': f"336x336",
            'face_num': len(faces)
        })
    new_df = pd.DataFrame(new_rows)
    new_csv_path = os.path.join(save_dir, os.path.basename(csv_path))
    new_df.to_csv(new_csv_path, index=False)

if __name__ == "__main__":
    main('train.csv', save_dir='../croped_data')
    main('val.csv', save_dir='../croped_data')
    main('test.csv', save_dir='../croped_data')
    val_df = pd.read_csv('../croped_data/val.csv')
    test_df = pd.read_csv('../croped_data/test.csv')
    val_test_df = pd.concat([val_df, test_df])
    val_test_df.to_csv('../croped_data/val_test.csv', index=False)
