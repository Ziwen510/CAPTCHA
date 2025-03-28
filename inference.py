import torch
import torch.nn as nn
from model import GateCNN
from PIL import Image
import glob
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import csv

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

    # Hyperparameters and model configuration
    num_gateblocks = 8
    input_channels = 3
    mid1_channels = 64
    mid2_channels = 512
    model_name = "GateCNN_v4_epoch15_gates8_64_512_channel3_lr0.001_batchsize8"

    # Instantiate the model and move it to the device
    model = GateCNN(num_classes=num_classes, num_gateblocks=num_gateblocks,
                    input_channels=input_channels, mid1_channels=mid1_channels,
                    mid2_channels=mid2_channels)
    model = nn.DataParallel(model)
    model = model.to(device)
    checkpoint = torch.load(f"checkpoints/{model_name}.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_folder = "dataset//test_cleaned_color_resized"
    dataset = OCRDataset(folder_path=test_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []  # to store each row as [ground_truth, prediction]
    correct = 0
    total = 0

    with torch.no_grad():
        for image, image_path in dataloader:
            image = image.to(device) 
            logits = model(image)  # (B, num_classes, T)
            logits = logits.squeeze(0)  # (num_classes, T)
            pred_str = ctc_greedy_decoder(logits, blank=0)

            # Extract ground truth from the file name (everything before "-0.png")
            base_name = os.path.basename(image_path[0])
            if "-0.png" in base_name:
                ground_truth = base_name.split("-0.png")[0]
            else:
                ground_truth = base_name  # fallback if pattern not found

            results.append([ground_truth, pred_str])
            total += 1
            if pred_str == ground_truth:
                correct += 1

            print(f"{ground_truth} -> {pred_str}")

    accuracy = correct / total if total > 0 else 0.0
    # Append a final row with the accuracy
    results.append(["Accuracy", f"{accuracy:.4f}"])

    # Write the results to a CSV file
    csv_file = f"test_results/test_results_{model_name}.csv"
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ground Truth", "Prediction"])
        for row in results:
            writer.writerow(row)

    print(f"Results written to {csv_file}")

if __name__ == "__main__":
    run_inference()
