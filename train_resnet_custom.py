import os
import sys
import time # TAMBAHAN: Untuk menghitung waktu training
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from utils.datasets import CustomImageDataset

# Gunakan GPU jika tersedia, jika tidak gunakan CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/caltech101" # Sesuaikan dengan jalur folder Anda
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10

# =========================================================================
# ARSITEKTUR CUSTOM RESNET (BUATAN SENDIRI, BUKAN DARI LIBRARY)
# =========================================================================
class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Jalan pintas (Skip Connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity # Ini ciri khas ResNet: menambahkan input asli ke output
        out = self.relu(out)
        return out

class CustomResNet(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = SimpleResidualBlock(64, 64, stride=1)
        self.layer2 = SimpleResidualBlock(64, 128, stride=2)
        self.layer3 = SimpleResidualBlock(128, 256, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
# =========================================================================

# Persiapkan transformasi gambar (Sama persis dengan Base AlexNet)
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

# Gunakan seed agar pembagian data 100% sama dengan saat ngetes AlexNet
train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# Inisialisasi model CustomResNet
model = CustomResNet(num_classes=len(full_dataset.classes)).to(DEVICE)
criterion = nn.CrossEntropyLoss() # fungsi loss
optimizer = optim.Adam(model.parameters(), lr=LR)

print("Memulai pelatihan model Custom ResNet...")
start_time = time.time() # Mulai catat waktu

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
            # Hitung Loss Validasi
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            
            # Hitung Akurasi Validasi
            preds = torch.max(outputs, 1)[1]
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
            
    val_loss = val_running_loss / len(val_loader.dataset)
    val_acc = correct_val / total_val
    
    # Cetak 4 Metrik Sekaligus
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

end_time = time.time() # Selesai catat waktu
total_waktu = (end_time - start_time) / 60

print(f"\nSelesai! Total waktu training Custom ResNet: {total_waktu:.2f} menit")

# ==========================================
# PENYIMPANAN MODEL
# ==========================================
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'training_result/resnet_custom.pth')
print("Model berhasil disimpan ke 'training_result/resnet_custom.pth'.")