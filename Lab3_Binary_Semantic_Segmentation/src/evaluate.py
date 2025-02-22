import torch
from tqdm import tqdm
from models.unet import UNet
from models.resnet34_unet import UNetResNet34
from oxford_pet import load_dataset
from utils import dice_score, load_checkpoint, get_device

def evaluate_model(model, data_loader, device):
    model.eval()
    total_dice = 0.0

    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            total_dice += dice_score(pred, masks)

    avg_dice = total_dice / len(data_loader)
    return avg_dice

if __name__ == "__main__":
    data_path = "../dataset"
    test_loader = load_dataset(data_path, "test")
    device = get_device()

    #需要將pth的檔名改成best_model_UNet.pth
    unet_model = UNet(n_channels=3, n_classes=2)
    _, _ = load_checkpoint("../saved_models/best_model_UNet.pth", unet_model)
    unet_model = unet_model.to(device)
    unet_dice = evaluate_model(unet_model, test_loader, device)

    print(f"UNet Dice Score on Test Set: {unet_dice:.4f}")

    #需要將pth的檔名改成best_model_UNetResNet34.pth
    resnet_unet_model = UNetResNet34(num_classes=2)
    _, _ = load_checkpoint("../saved_models/best_model_UNetResNet34.pth", resnet_unet_model)
    resnet_unet_model = resnet_unet_model.to(device)
    resnet_unet_dice = evaluate_model(resnet_unet_model, test_loader, device)
    print(f"ResNet34-UNet Dice Score on Test Set: {resnet_unet_dice:.4f}")