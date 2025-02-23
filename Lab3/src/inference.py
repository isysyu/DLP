import torch
from PIL import Image
import torchvision.transforms as transforms
from models.unet import UNet
from models.resnet34_unet import UNetResNet34
from utils import visualize_prediction, load_checkpoint, get_device


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def inference(model, image_path, device):
    model.eval()
    image = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1)

    return image.squeeze().cpu(), prediction.squeeze().cpu()


if __name__ == "__main__":
    device = get_device()

    unet_model = UNet(n_channels=3, n_classes=2)
    _, _ = load_checkpoint("../saved_models/best_model_UNet.pth", unet_model)
    unet_model = unet_model.to(device)

    resnet_unet_model = UNetResNet34(num_classes=2)
    _, _ = load_checkpoint("../saved_models/best_model_UNetResNet34.pth", resnet_unet_model)
    resnet_unet_model = resnet_unet_model.to(device)

    image_path = "../dataset/oxford-iiit-pet/images/yorkshire_terrier_102.jpg"

    unet_image, unet_prediction = inference(unet_model, image_path, device)
    resnet_unet_image, resnet_unet_prediction = inference(resnet_unet_model, image_path, device)

    visualize_prediction(unet_image, torch.zeros_like(unet_prediction), unet_prediction)
    visualize_prediction(resnet_unet_image, torch.zeros_like(resnet_unet_prediction), resnet_unet_prediction)