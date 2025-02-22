import torch
import torch.nn as nn
import torch.optim as optim
from SCCNet import SCCNet
from Dataloader import get_dataloader

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.2f}%')

    return model

def train_SD(num_epochs=50, learning_rate=0.001, batch_size=32):
    model = SCCNet(Nu=22, Nc=20, Nt=1).to(device)
    train_loader = get_dataloader('train', batch_size=batch_size)
    val_loader = get_dataloader('test', batch_size=batch_size)
    return train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)

def train_LOSO(num_epochs=50, learning_rate=0.001, batch_size=32):
    model = SCCNet(Nu=22, Nc=20, Nt=1).to(device)
    train_loader = get_dataloader('train', batch_size=batch_size)
    val_loader = get_dataloader('test', batch_size=batch_size)
    return train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)

def train_LOSO_FT(num_epochs_initial=50, num_epochs_ft=20, learning_rate=0.001, batch_size=32):
    model = SCCNet(Nu=22, Nc=20, Nt=1).to(device)
    train_loader = get_dataloader('train', batch_size=batch_size)
    val_loader = get_dataloader('test', batch_size=batch_size)
    model = train_model(model, train_loader, val_loader, num_epochs_initial, learning_rate, device)

    # Fine-tuning
    ft_loader = get_dataloader('finetune', batch_size=batch_size)
    return train_model(model, ft_loader, val_loader, num_epochs_ft, learning_rate * 0.1, device)

train_SD(num_epochs=50, learning_rate=0.001, batch_size=32)