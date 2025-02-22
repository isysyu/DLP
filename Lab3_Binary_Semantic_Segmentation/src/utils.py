import torch
import matplotlib.pyplot as plt
import numpy as np
import logging
import os


def dice_score(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)


def visualize_prediction(image, mask, prediction):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(image.permute(1, 2, 0))
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(mask.squeeze(), cmap='gray')
    ax2.set_title('Ground Truth')
    ax2.axis('off')

    ax3.imshow(prediction.squeeze(), cmap='gray')
    ax3.set_title('Prediction')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()


def save_checkpoint(state, filename="../saved_models/checkpoint.pth"):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename, model, optimizer=None):
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'], checkpoint['best_dice_score']


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    # 添加控制台輸出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def plot_learning_curve(train_dice_history, val_dice_history, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot([tensor.cpu().numpy() for tensor in train_dice_history], label='Train Dice')
    plt.plot([tensor.cpu().numpy() for tensor in val_dice_history], label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.savefig(save_path)
    plt.close()