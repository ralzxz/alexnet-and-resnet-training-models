import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
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

# =======================================================
# TAMBAHAN: Memisahkan transformasi Data Latih & Validasi
# =======================================================
# Transformasi khusus Data Latih (Dengan Augmentasi)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), # Tambahan: Membalik gambar 50%
    transforms.RandomRotation(15),          # Tambahan: Memutar gambar acak max 15 derajat
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Transformasi khusus Data Validasi (Murni, tanpa augmentasi)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# TAMBAHAN: Muat dataset dua kali dengan transform yang berbeda
dataset_train = CustomImageDataset(root_dir=DATA_DIR, transform=train_transforms)
dataset_val = CustomImageDataset(root_dir=DATA_DIR, transform=val_transforms)

# Bagi dataset: 80% untuk latihan, 20% untuk validasi
train_size = int(0.8 * len(dataset_train))
val_size = len(dataset_train) - train_size

# TAMBAHAN: Gunakan manual_seed agar pembagiannya konsisten
train_set, val_set = random_split(dataset_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# TAMBAHAN TRIK: Ganti sumber data validasi agar memakai dataset yang murni
val_set.dataset = dataset_val

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# Inisialisasi model, fungsi loss, dan optimizer
model = AlexNetCaltech(num_classes=len(dataset_train.classes)).to(DEVICE)
criterion = nn.CrossEntropyLoss() # fungsi loss
optimizer = optim.Adam(model.parameters(), lr=LR)

# =======================================================
# TAMBAHAN: Inisialisasi Learning Rate Scheduler
# =======================================================
scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

print("Memulai pelatihan model yang di-Improve (Augmentasi + Scheduler)...")

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
            
        train_loss = running_loss / len(train_set) # TAMBAHAN: Disesuaikan jadi train_set
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
                
        val_loss = val_running_loss / len(val_set) # TAMBAHAN: Disesuaikan jadi val_set
        val_acc = correct_val / total_val
        
        # =======================================================
        # TAMBAHAN: Majukan step scheduler dan ambil nilai LR
        # =======================================================
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Cetak 4 Metrik Sekaligus + Learning Rate!
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# ==========================================
# PENYIMPANAN MODEL
# ==========================================
os.makedirs('models', exist_ok=True)
# TAMBAHAN: Ubah nama file sedikit agar model aslimu tidak tertimpa
torch.save(model.state_dict(), 'training_result/alexnet_caltech_improved.pth')
print("\nProses pelatihan selesai. Model berhasil disimpan ke 'training_result/alexnet_caltech_improved.pth'.")