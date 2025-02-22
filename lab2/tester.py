import torch
from SCCNet import SCCNet
from Dataloader import get_dataloader

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def test_model(model_path, batch_size=32):
    model = SCCNet(Nu=22, Nc=20, Nt=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loader = get_dataloader('test', batch_size=batch_size)
    return test(model, test_loader, device)