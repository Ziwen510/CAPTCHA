import torch
import torch.nn as nn
from gateCNN import GateCNN 
from baseCNN import BaseCNN
from CRNN import CRNN
from PIL import Image
import glob
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import csv
import concurrent.futures

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

def run_inference_for_epoch(epoch, gpu_id):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Inference on device: {device} for epoch: {epoch}")

    # Hyperparameters
    input_channels = 3
    hidden_channels = 128
    pretrained = True
    backbone = "resnet50"
    num_lstm_layers = 4
    # model_name = f"CRNN_epoch{epoch}_resnet50_True_lstmhidden256_lstmlayer4_channel3_lr0.001_batchsize64"
    model_name = f"BaseCNN_epoch{epoch}_hidden128_channel3_lr0.001_batchsize16"

    # Instantiate the model
    model = BaseCNN(num_classes=num_classes, hidden_channels=hidden_channels)
    # model = CRNN(num_chars=num_classes, hidden_size=hidden_channels, backbone=backbone,
    #              pretrained=pretrained, num_lstm_layers=num_lstm_layers)
    model = model.to(device)

    # Load the checkpoint for the current epoch
    checkpoint_path = f"checkpoints/{model_name}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    # Remove "module." prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_folder = "dataset//test_cleaned_color_resized"
    dataset = OCRDataset(folder_path=test_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []
    correct = 0
    total = 0

    with torch.no_grad():
        for image, image_path in dataloader:
            image = image.to(device)
            logits = model(image)  # (B, num_classes, T)
            logits = logits.squeeze(0)  # (num_classes, T)
            pred_str = ctc_greedy_decoder(logits, blank=0)

            base_name = os.path.basename(image_path[0])
            if "-0.png" in base_name:
                ground_truth = base_name.split("-0.png")[0]
            else:
                ground_truth = base_name

            results.append([ground_truth, pred_str])
            total += 1
            if pred_str == ground_truth:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Epoch {epoch} Accuracy: {accuracy:.4f}")
    results.append(["Accuracy", f"{accuracy:.4f}"])

    # Write the results to a CSV file
    csv_file = f"test_results/test_results_{model_name}.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ground Truth", "Prediction"])
        for row in results:
            writer.writerow(row)

    print(f"Results written to {csv_file}")
    return accuracy

def main():
    epochs = range(10, 26)
    accuracies = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        future_to_epoch = {}
        for i, epoch in enumerate(epochs):
            gpu_id = i % 8 
            future = executor.submit(run_inference_for_epoch, epoch, gpu_id)
            future_to_epoch[future] = epoch

        for future in concurrent.futures.as_completed(future_to_epoch):
            epoch = future_to_epoch[future]
            try:
                acc = future.result()
                accuracies[epoch] = acc
            except Exception as exc:
                print(f"Epoch {epoch} generated an exception: {exc}")

    for epoch in sorted(accuracies):
        print(f"Epoch {epoch}: Accuracy = {accuracies[epoch]:.4f}")

if __name__ == "__main__":
    main()
