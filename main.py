import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime

from model.model import build_model
from dataset.dataset import build_dataloader, InputNormalize, build_finetune_dataloader
from core.train_utils import (
    train_one_epoch, extract_features, inference
)
from core.utils import setup_logger, set_seed, get_minimize_lr_epochs
from loss.custom_loss import FocalLoss
from pathlib import Path




def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch 6-Class Classification with ResNet')
    parser.add_argument('--train_data_file', type=str, default='prepare_data/raw_data/train.csv', help='Path to data directory')
    parser.add_argument('--test_data_file', type=str, default='prepare_data/raw_data/val_test.csv', help='Path to test_data_file')
    parser.add_argument('--val_data_file', type=str, default='prepare_data/raw_data/val.csv', help='Path to val_data_file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='./results/exp1', help='Directory to save results')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'extract_features', 'generate_data', 'finetune'], help='Mode: train or test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--model', type=str, default='resnet18', help='model')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'focal'], help='loss function')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingWarmRestarts', help='pretrained')
    parser.add_argument('--T_0', type=int, default=10)
    parser.add_argument('--T_mult', type=int, default=2)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--save_every_epochs', type=int, default=50, help='Save model every n epochs if using ReduceLROnPlateau scheduler')
    parser.add_argument('--use_mixup', action='store_true', help='Use mixup data augmentation')
    parser.add_argument('--oversampling', action='store_true', help='Use oversampling data augmentation')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone layers for fine-tuning')
    parser.add_argument('--remove_illegal_faces', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)


    # 设置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.save_dir, f"{args.mode}_{timestamp}.log")
    logger = setup_logger(log_path)
    
    logger.info("启动程序，参数如下：")
    for arg in vars(args):
        logger.info(f" {arg}: {getattr(args, arg)}")
    
    # 设置随机种子以确保结果可复现
    set_seed(args.seed, logger)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    input_normalizer = InputNormalize(device=device)

    

    # Get model
    model = build_model(args.model, num_classes=args.num_classes)
    model = model.to(device)
    
    # Define loss function and optimizer
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss(gamma=args.focal_gamma)
    else:
        raise NotImplementedError('ce, focal loss are supported only')
    
    
    

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint if 'model_state_dict' not in checkpoint else checkpoint['model_state_dict'])
        else:
            logger.warning(f"未找到检查点: {args.resume}")
    
    # Training mode
    if args.mode == 'train':

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # Learning rate scheduler
        if args.scheduler == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        elif args.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.T_0, args.T_mult)
        else:
            raise NotImplementedError('ReduceLROnPlateau, CosineAnnealingWarmRestarts are supported only')
        
        # Get data loaders
        train_data_loader, train_face_dataset = build_dataloader(
            args.train_data_file, 
            is_train=True, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            seed=args.seed,
            shuffle=True,
            oversampling=args.oversampling,
            remove_illegal_faces=args.remove_illegal_faces
        )
        logger.info(f"数据加载器已创建，类别顺序为{train_face_dataset.label2classid}, 共有{len(train_face_dataset)}个样本")
        
        test_data_loader1, test_face_dataset1 = build_dataloader(
            args.test_data_file, 
            is_train=False, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            seed=args.seed,
            shuffle=False,
            use_tta=True
        )

        test_data_loader2, test_face_dataset2 = build_dataloader(
            args.test_data_file, 
            is_train=False, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            seed=args.seed,
            shuffle=False,
            use_tta=False
        )

        # Lists to store metrics
        train_losses = []
        train_accs = []
        train_loss, train_acc = 0, 0
        # Training loop
        for epoch in range(args.num_epochs):
            logger.info(f"\n第 {epoch+1}/{args.num_epochs} 轮训练")
            
            # Train
            train_loss, train_acc = train_one_epoch(model, train_data_loader, criterion, optimizer, device, input_normalizer, args.use_mixup)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            logger.info(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"当前学习率: {current_lr:.6f}")

            # Update learning rate
            if args.scheduler == 'ReduceLROnPlateau':
                scheduler.step(train_loss)
            else:
                scheduler.step()
            

            # Save checkpoint every 50 epochs if use ReduceLROnPlateau
            if ((epoch + 1) % args.save_every_epochs == 0) and args.scheduler == "ReduceLROnPlateau":
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth'))
                logger.info(f"保存第 {epoch+1} 轮模型，训练准确率: {train_acc:.2f}%")
            
            if args.scheduler == "CosineAnnealingWarmRestarts":
                save_epochs = get_minimize_lr_epochs(args.T_0, args.T_mult)
                if (epoch + 1) in save_epochs:
                    torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth'))
                    logger.info(f"保存第 {epoch+1} 轮模型，训练准确率: {train_acc:.2f}%")
                    save_dir1 = Path(args.save_dir) / f"test_epoch{epoch+1}+withTTA"
                    inference(model, test_data_loader1, device, save_dir1, input_normalizer)
                    save_dir2 = Path(args.save_dir) / f"test_epoch{epoch+1}+withoutTTA"
                    inference(model, test_data_loader2, device, save_dir2, input_normalizer)
        
        logger.info("训练完成!")
    
    elif args.mode == 'extract_features':
        save_dir = Path(args.resume).parent / f'{Path(args.resume).stem}_extracted_features_{Path(args.train_data_file).stem}'
        data_loader, _ = build_dataloader(
            args.train_data_file, 
            is_train=False, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            seed=args.seed,
            shuffle=False,
            use_tta=False
        )
        extract_features(model, data_loader, save_dir, device, input_normalizer)


    elif args.mode == 'inference':
        test_data_loader1, _ = build_dataloader(
            args.test_data_file, 
            is_train=False, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            seed=args.seed,
            shuffle=False,
            use_tta=True
        )

        test_data_loader2, _ = build_dataloader(
            args.test_data_file, 
            is_train=False, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            seed=args.seed,
            shuffle=False,
            use_tta=False
        )
        save_dir1 = Path(args.resume).parent / f"{Path(args.resume).stem}_inference_withTTA"
        save_dir2 = Path(args.resume).parent / f"{Path(args.resume).stem}_inference_withoutTTA"
        inference(model, test_data_loader1, device, save_dir1, input_normalizer)
        inference(model, test_data_loader2, device, save_dir2, input_normalizer)


    else:
        raise ValueError("illegal mode")

if __name__ == '__main__':
    main() 