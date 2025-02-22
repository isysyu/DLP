import matplotlib.pyplot as plt
import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()

def plot_accuracy(accuracies, methods):
    plt.figure(figsize=(10, 6))
    plt.bar(methods, accuracies)
    plt.title('Accuracy Comparison')
    plt.xlabel('Training Methods')
    plt.ylabel('Accuracy (%)')
    plt.savefig('accuracy_comparison.png')
    plt.close()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
    return model