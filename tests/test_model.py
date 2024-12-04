import torch
import pytest
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MNISTModel
from torchvision import datasets, transforms

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = MNISTModel()
    assert count_parameters(model) < 100000, "Model has too many parameters"

def test_strict_parameter_limit():
    model = MNISTModel()
    param_count = count_parameters(model)
    assert param_count < 100000, f"Model has {param_count} parameters, which exceeds the strict limit of 100,000"

def test_layer_sizes():
    model = MNISTModel()
    assert model.conv1.out_channels == 8, "First conv layer should have 8 output channels"
    assert model.conv2.out_channels == 16, "Second conv layer should have 16 output channels"
    assert model.fc1.out_features == 64, "First FC layer should output 64 features"
    assert model.fc2.out_features == 10, "Final layer should output 10 classes"

def test_input_output_shape():
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape is incorrect"

def test_model_accuracy():
    # Load the latest trained model
    model = MNISTModel()
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
    model.load_state_dict(torch.load(os.path.join(models_dir, latest_model)))
    
    # Prepare test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 80, f"Model accuracy {accuracy:.2f}% is below 80%"