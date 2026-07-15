import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from models.alexnet import AlexNetCaltech
from utils.datasets import CustomImageDataset

# Gunakan GPU jika tersedia, jika tidak gunakan CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/caltech101" # Sesuaikan dengan jalur folder Anda
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10

# Persiapkan transformasi gambar
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Muat seluruh dataset
full_dataset = CustomImageDataset(root_dir=DATA_DIR, transform=data_transforms)

# Bagi dataset: 80% untuk latihan, 20% untuk validasi
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# Inisialisasi model, fungsi loss, dan optimizer
model = AlexNetCaltech(num_classes=len(full_dataset.classes)).to(DEVICE)
criterion = nn.CrossEntropyLoss() # fungsi loss
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
        # =========================
        # 1. TAHAP PELATIHAN (TRAIN)
        # =========================
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Hitung Loss Training
            running_loss += loss.item() * images.size(0)
            
            # Hitung Akurasi Training
            preds = torch.max(outputs, 1)[1]
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train

        # =========================
        # 2. TAHAP VALIDASI (VAL)
        # =========================
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(images)
                # Hitung Loss Validasi (Tambahan baru)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                
                # Hitung Akurasi Validasi
                preds = torch.max(outputs, 1)[1]
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = correct_val / total_val
        
        # Cetak 4 Metrik Sekaligus!
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# ==========================================
# PENYIMPANAN MODEL
# ==========================================
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'training_result/alexnet_caltech.pth')
print("\nProses pelatihan selesai. Model berhasil disimpan ke 'training_result/alexnet_caltech.pth'.")