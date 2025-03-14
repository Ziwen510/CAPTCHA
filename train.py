import torch
import torch.nn as nn
import torch.optim as optim
from model import GateCNN
from PIL import Image
import glob
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

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

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)

    num_gateblocks = 4
    input_channels = 3
    num_epochs = 10
    batch_size = 2
    learning_rate = 1e-3

    model = GateCNN(num_classes=num_classes, num_gateblocks=num_gateblocks, input_channels=input_channels)
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    transform = transforms.Compose([
        # transforms.Resize((32, 128)),
        transforms.ToTensor()
    ])

    dataset = OCRDataset(folder_path="dataset//train_cleaned_color_resized", transform=transform, char_to_idx=char_to_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=ocr_collate_fn)

    for epoch in range(num_epochs):
        count = 0
        for images, targets_concat, target_lengths in dataloader:
            count = count + 1
            images = images.to(device)
            targets_concat = targets_concat.to(device)
            target_lengths = target_lengths.to(device)

            probs, logits = model(images) # (B, num_classes, W)
            # CTCLoss expects logits with shape (T, B, C), where T = time steps.
            logits_ctc = logits.permute(2, 0, 1)
            batch_size_actual = images.size(0)
            input_lengths = torch.full(size=(batch_size_actual,), fill_value=logits_ctc.size(0), dtype=torch.long).to(device)

            loss = ctc_loss_fn(logits_ctc.log_softmax(2), targets_concat, input_lengths, target_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}/{num_epochs} Batch {count}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "gatecnn_model.pth")
    print("Model saved as gatecnn_model.pth")

if __name__ == "__main__":
    train_model()
