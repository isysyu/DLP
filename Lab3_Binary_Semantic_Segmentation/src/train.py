import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from models.unet import UNet
from models.resnet34_unet import UNetResNet34
from oxford_pet import load_dataset
from utils import dice_score, save_checkpoint, setup_logger, plot_learning_curve

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, logger=None):
    device = get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_dice_score = 0.0
    train_dice_history = []
    val_dice_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            train_dice += dice_score(pred, masks)

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        val_loss, val_dice = evaluate_model(model, val_loader, criterion, device)

        train_dice_history.append(train_dice.cpu())
        val_dice_history.append(val_dice.cpu())

        if logger:
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        if val_dice > best_dice_score:
            best_dice_score = val_dice
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dice_score': best_dice_score,
            }, filename=f"../saved_models/best_model_{type(model).__name__}.pth")
            if logger:
                logger.info(f"New best model saved with Dice score: {best_dice_score:.4f}")

    return model, train_dice_history, val_dice_history

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            val_dice += dice_score(pred, masks)

    val_loss /= len(data_loader)
    val_dice /= len(data_loader)
    return val_loss, val_dice

if __name__ == "__main__":
    data_path = "../dataset"
    batch_size = 2
    train_loader = load_dataset(data_path, "train", batch_size=batch_size)
    val_loader = load_dataset(data_path, "valid", batch_size=batch_size)


    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger('train_log', f'{log_dir}/train.log')


    device = get_device()
    logger.info(f"Using device: {device}")

    # training UNet
    logger.info("Starting UNet training")
    unet_model = UNet(n_channels=3, n_classes=2)
    unet_model, unet_train_history, unet_val_history = train_model(unet_model, train_loader, val_loader, num_epochs=50, learning_rate=0.0006, logger=logger)
    plot_learning_curve(unet_train_history, unet_val_history, f'{log_dir}/unet_learning_curve.png')

    # training resnet34_unet
    #logger.info("Starting ResNet34-UNet training")
    #resnet_unet_model = UNetResNet34(num_classes=2)
    #resnet_unet_model, resnet_train_history, resnet_val_history = train_model(resnet_unet_model, train_loader, val_loader, num_epochs=50, learning_rate=1e-3, logger=logger)
    #plot_learning_curve(resnet_train_history, resnet_val_history, f'{log_dir}/resnet_unet_learning_curve.png')

    logger.info("Training completed")