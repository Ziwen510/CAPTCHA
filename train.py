import torch
import torch.nn as nn
import torch.optim as optim
from model import GateCNN 
from PIL import Image
import glob
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import csv
import argparse

characters = "abcdefghijklmnopqrstuvwxyz0123456789"
char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
num_classes = 37

class OCRDataset(Dataset):
    def __init__(self, folder_path, transform=None, char_to_idx=None):
        self.image_files = glob.glob(os.path.join(folder_path, "*.png"))
        self.transform = transform
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        base_name = os.path.basename(image_path)
        name_no_ext = os.path.splitext(base_name)[0]
        if "-" in name_no_ext:
            ground_truth = name_no_ext.split("-")[0]
        else:
            ground_truth = name_no_ext
        target = [self.char_to_idx[c] for c in ground_truth if c in self.char_to_idx]

        return image, torch.tensor(target, dtype=torch.long)

def ocr_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets)
    return images, targets_concat, target_lengths

def train_model(resume_checkpoint=None, extra_epochs=5, start_epoch=5):
    # Select device (ensure CUDA is available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)

    # Hyperparameters and model configuration
    model_name = "GateCNN_v4"
    num_gateblocks = 8
    input_channels = 3
    # Base number of epochs if training from scratch
    base_epochs = 5
    batch_size = 8
    learning_rate = 1e-3
    mid1_channels = 64
    mid2_channels = 512

    # Instantiate the model and move it to the device
    model = GateCNN(num_classes=num_classes, num_gateblocks=num_gateblocks,
                    input_channels=input_channels, mid1_channels=mid1_channels,
                    mid2_channels=mid2_channels)
    model = model.to(device)
    
    # If multiple GPUs are available, use DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    transform = transforms.Compose([
        # transforms.Resize((32, 128)),
        transforms.ToTensor()
    ])

    dataset = OCRDataset(folder_path="dataset/train_cleaned_color_resized",
                         transform=transform, char_to_idx=char_to_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=ocr_collate_fn)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # If resuming from a checkpoint, load the model and optimizer states and set the start epoch.
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint)
        total_epochs = start_epoch + extra_epochs
        print(f"Resuming training from epoch {start_epoch} for {extra_epochs} additional epochs (total epochs: {total_epochs}).")
    else:
        if resume_checkpoint is not None:
            print(f"Checkpoint file {resume_checkpoint} not found. Starting training from scratch.")
        start_epoch = 0
        total_epochs = base_epochs

    for epoch in range(start_epoch, total_epochs):
        losses_log = []
        count = 0
        for images, targets_concat, target_lengths in dataloader:
            count += 1
            images = images.to(device)
            targets_concat = targets_concat.to(device)
            target_lengths = target_lengths.to(device)

            # Forward pass
            logits = model(images)  # (B, num_classes, W)
            # For CTCLoss, permute logits to (T, B, C)
            logits_ctc = logits.permute(2, 0, 1)
            batch_size_actual = images.size(0)
            input_lengths = torch.full(size=(batch_size_actual,), fill_value=logits_ctc.size(0), dtype=torch.long).to(device)

            loss = ctc_loss_fn(logits_ctc.log_softmax(2), targets_concat, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{total_epochs} Batch {count}, Loss: {loss.item():.4f}")
            losses_log.append([epoch+1, count, loss.item()])

        # Write loss log to a CSV file after each epoch
        csv_filename = f"losses_{model_name}_epoch{epoch+1}_gates{num_gateblocks}_{mid1_channels}_{mid2_channels}_channel{input_channels}_lr{learning_rate}_batchsize{batch_size}.csv"
        csv_path = os.path.join(checkpoint_dir, csv_filename)
        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Epoch", "Batch", "Loss"])
            writer.writerows(losses_log)
        print(f"Loss log for epoch {epoch+1} saved at {csv_path}")

        # Save a checkpoint after the end of each epoch
        checkpoint_filename = f"{model_name}_epoch{epoch+1}_gates{num_gateblocks}_{mid1_channels}_{mid2_channels}_channel{input_channels}_lr{learning_rate}_batchsize{batch_size}.pth"
        checkpoint_path_epoch = os.path.join(checkpoint_dir, checkpoint_filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path_epoch)
        print(f"Checkpoint for epoch {epoch+1} saved at {checkpoint_path_epoch}")

if __name__ == "__main__":
    resume_checkpoint = "checkpoints/GateCNN_v4_epoch5_gates8_64_512_channel3_lr0.001_batchsize8.pth"
    extra_epochs = 10
    start_epoch = 5
    train_model(resume_checkpoint, extra_epochs, start_epoch)
