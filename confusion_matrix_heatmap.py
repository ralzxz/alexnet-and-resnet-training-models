import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.alexnet import AlexNetCaltech
from utils.datasets import CustomImageDataset

# 1. Konfigurasi
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/caltech101"

# 2. Siapkan Data Validasi
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = CustomImageDataset(root_dir=DATA_DIR, transform=data_transforms)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

_, val_set = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# 3. Muat Model Buatanmu
model = AlexNetCaltech(num_classes=len(full_dataset.classes)).to(DEVICE)
model.load_state_dict(torch.load('training_result/alexnet_caltech_result.pth', map_location=DEVICE))
model.eval()

# 4. Kumpulkan Prediksi
semua_prediksi = []
semua_label = []

print("Sedang memproses gambar untuk Confusion Matrix (tunggu sebentar)...")
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = torch.max(outputs, 1)[1]
        
        semua_prediksi.extend(preds.cpu().numpy())
        semua_label.extend(labels.cpu().numpy())

# 5. BUAT GRAFIK CONFUSION MATRIX (SEABORN)
print("Menggambar grafik...")
cm = confusion_matrix(semua_label, semua_prediksi)

# Kita buat kanvas yang SANGAT BESAR (24x20) karena ada 101 kelas
plt.figure(figsize=(24, 20))

# annot=False agar angkanya tidak ditulis di dalam kotak (terlalu sempit untuk 101 kelas)
# cmap='Blues' memberikan warna biru, semakin gelap = semakin banyak tebakan yang benar
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=5, yticklabels=5)

plt.title('Confusion Matrix - AlexNet pada Caltech-101', fontsize=24, pad=20)
plt.xlabel('Prediksi Model (Indeks Kelas)', fontsize=18, labelpad=15)
plt.ylabel('Label Asli (Indeks Kelas)', fontsize=18, labelpad=15)

# --- PASTIKAN ROTASINYA JADI 0 AGAR ANGKA TIDAK MIRING ---
plt.xticks(rotation=0, fontsize=12) 
plt.yticks(rotation=0, fontsize=12)

plt.tight_layout()
plt.savefig('grafik_confusion_matrix.png', dpi=300) # Resolusi tinggi (300 dpi) agar tidak pecah saat di-zoom
print("Selesai! Grafik berhasil disimpan sebagai 'confusion_matrix_graph.png'.")

plt.show()