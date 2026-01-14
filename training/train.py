import torch
import os
from torch.utils.data import DataLoader
from training.dataset_loader import TongueDataset
from training.unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TongueDataset(
    "data/dataset",
    "data/groundtruth/mask"
)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UNet().to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

os.makedirs("model", exist_ok=True)

for epoch in range(10):
    for img, mask in loader:
        img = img.to(device)
        mask = mask.to(device)
        
        pred = model(img)
        loss = criterion(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss: {loss.item()}")

torch.save(model.state_dict(), "model/tongue_unet.pth")
