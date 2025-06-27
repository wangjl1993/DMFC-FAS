import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
from torchvision.utils import save_image
from core.utils import compute_similarity
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam



def train_one_epoch(model, dataloader, criterion, optimizer, device, input_normalizer, use_mixup=False):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for (inputs, labels, _, _) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()
        
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.4)
            outputs = model(input_normalizer(inputs))
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(input_normalizer(inputs))
            loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, input_normalizer):
    """
    Validate the model
    
    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run the validation on
        logger: 日志记录器
        
    Returns:
        tuple: (validation loss, validation accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for (inputs, labels, _, _) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(input_normalizer(inputs))
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.mean().item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.mean().item(), 'acc': 100 * correct / total})
    
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def test(model, 
        test_loader, 
        device, 
        logger, 
        save_dir, 
        input_normalizer,
        label2classid):
    """
    在 test.csv 上推理，并将类别1的分数保存到 test_scores.txt
    """

    live_classid = 1 #label2classid['Live Face']
    model.eval()
    img_name_score = {}
    with torch.no_grad():
        for (inputs, _, _, img_name)  in tqdm(test_loader, desc="Testing"):
            
            inputs = inputs.to(device)
            outputs = model(input_normalizer(inputs))
            probs = F.softmax(outputs, dim=1)[:, live_classid]

            for i in range(len(img_name)):
                img_name_score[img_name[i]] = probs[i].item()
    # 保存分数
    with open(os.path.join(save_dir, 'test_scores.txt'), 'w') as f:
        for img_name, score in img_name_score.items():
            score = max(0, min(1, score))
            f.write(f"{img_name} {1 - score}\n")
    logger.info(f'Test scores saved to {os.path.join(save_dir, "test_scores.txt")}')
    
def inference(model, dataloader, device, save_dir, input_normalizer):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    extracted_features, df = extract_features(model, dataloader, save_dir, device, input_normalizer)

    val_live_mask = (df['label'] == 5) & (df['filename'].str.startswith('Data-val')) # liveface classid = 5
    val_live_features = extracted_features[val_live_mask].mean(0)

    scores = compute_similarity(extracted_features, val_live_features, method='cosine').reshape(-1)
    save_file = save_dir / 'scores.txt'
    with open(save_file, 'w') as f:
        for file, score in zip(df['filename'], scores):
            score = max(0, min(1, score))
            f.write(f"{file} {1-score}\n")

    
    

def extract_features(model, dataloader, save_dir, device, input_normalizer):

    model.eval()
    features_list = []
    labels_list  = []
    filenames_list = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="extract features")
        for (inputs, labels, _, filenames) in pbar:
            inputs = inputs.to(device)
            outputs = model.extract_features(input_normalizer(inputs)).cpu().numpy()

            features_list.append(outputs)
            labels_list.append(labels.numpy())
            filenames_list.extend(filenames)
            
    features_list = np.concatenate(features_list, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    filenames_list = np.array(filenames_list)

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    np.save(save_dir / "features.npy", features_list)
    # np.save(save_dir / "labels.npy", labels_list)
    # np.save(save_dir / "filenames.npy", filenames_list)
    df = pd.DataFrame({"filename": filenames_list, "label": labels_list})
    df.to_csv(save_dir / "filenames_labels.csv", index=False)

    return features_list, df
           
def random_perturb(inputs, attack, eps):
    if attack == 'inf':
        r_inputs = 2 * (torch.rand_like(inputs) - 0.5) * eps
    else:
        r_inputs = (torch.rand_like(inputs) - 0.5).renorm(p=2, dim=1, maxnorm=eps)
    return r_inputs

def generate_data(model, dataloader, criterion, device, input_normalizer, save_dir, max_iter=6):

    model.eval()

    pbar = tqdm(dataloader, desc="generate data")
    for (inputs, labels, _, filenames) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        print(labels)
        random_noise = random_perturb(inputs, 'l2', 0.5)
        inputs = torch.clamp(inputs + random_noise, 0, 1).requires_grad_(True)
        optimizer = torch.optim.Adam([inputs], lr=1e-4)

        for iter in range(max_iter):
            
            optimizer.zero_grad()
            outputs = model(input_normalizer(inputs))
            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            with torch.no_grad():
                inputs.data.clamp_(0, 1)

            with torch.no_grad():
                generated_data = inputs.clone().detach()
                outputs = model(input_normalizer(generated_data))
                _, predicted = torch.max(outputs.data, 1)
                
                probs = F.softmax(outputs, dim=1)[:, labels[0]]
                # print(predicted, probs)
                if (predicted == labels).all():
                    save_file = Path(save_dir) / f"{iter}_{probs.item():.3f}_{filenames[0].replace('/', '_')}"
                    # print(f"Generated data saved to {save_file}")
                    save_image(generated_data.cpu(), save_file)
                    break



def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename, logger=None):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Validation loss
        accuracy: Validation accuracy
        filename: Path to save the checkpoint
        logger: 日志记录器
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filename)
    if logger:
        logger.info(f"检查点已保存到 {filename}")

def load_checkpoint(model, filename, optimizer=None, logger=None):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filename: Path to the checkpoint file
        logger: 日志记录器
        
    Returns:
        tuple: (model, optimizer, epoch, loss, accuracy)
    """
    
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:   
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        
        if logger:
            logger.info(f"已加载检查点，轮次: {epoch}，准确率: {accuracy:.2f}%")
    except Exception as e:
        model.load_state_dict(checkpoint)
        if logger:
            logger.info(f"已成功加载模型{filename}")
        optimizer, epoch, loss, accuracy = None, 0, 0, 0
    
    return model, optimizer, epoch, loss, accuracy 