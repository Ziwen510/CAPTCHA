import torch
import torch.nn as nn
from model import GateCNN
from PIL import Image
import glob
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

characters = "abcdefghijklmnopqrstuvwxyz0123456789"
char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
num_classes = 37
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

class OCRDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_files = glob.glob(os.path.join(folder_path, "*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path

# Greedy decoding function for CTC output.
def ctc_greedy_decoder(logits, blank=0):
    pred_indices = torch.argmax(logits, dim=0)  # shape (T,)
    pred_tokens = []
    previous = None
    for idx in pred_indices:
        idx_val = idx.item()
        if idx_val != previous and idx_val != blank:
            pred_tokens.append(idx_val)
        previous = idx_val
    pred_str = ''.join([idx_to_char[idx] for idx in pred_tokens])
    return pred_str

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Inference on device:", device)

    num_gateblocks = 4
    input_channels = 3
    model = GateCNN(num_classes=num_classes, num_gateblocks=num_gateblocks, input_channels=input_channels)
    model.load_state_dict(torch.load("gatecnn_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_folder = "dataset//test_cleaned_color_resized"
    dataset = OCRDataset(folder_path=test_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for image, image_path in dataloader:
            image = image.to(device) 
            _, logits = model(image) # (B, num_classes, T)
            logits = logits.squeeze(0)  # (num_classes, T)
            pred_str = ctc_greedy_decoder(logits, blank=0)
            print(f"Image: {os.path.basename(image_path[0])} -> Prediction: {pred_str}")

if __name__ == "__main__":
    run_inference()
