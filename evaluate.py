import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(
    "dataset/test",
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=1)

model = build_model()

model.load_state_dict(torch.load("resnet_xray_model.pth"))

model.to(device)

model.eval()

predictions = []
indices = []

with torch.no_grad():

    for i, (image, label) in enumerate(test_loader):

        image = image.to(device)

        output = model(image)

        pred = torch.argmax(output, dim=1)

        predictions.append(pred.item())
        indices.append(i)

df = pd.DataFrame({
    "image_index": indices,
    "predicted_label": predictions
})

df.to_csv("outputs/predictions.csv", index=False)

print("Predictions saved to outputs/predictions.csv")
