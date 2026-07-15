import matplotlib.pyplot as plt

# 1. MASUKKAN ANGKA ASLI DARI TERMINALMU DI SINI
epochs = list(range(1, 11))

# Train Loss dari Epoch 1 sampai 10
train_loss = [3.7517, 2.8792, 2.2674, 1.8504, 1.5239, 1.2054, 0.9379, 0.7165, 0.5152, 0.4025] 

# Train Acc dari Epoch 1 sampai 10
train_acc = [0.2060, 0.3706, 0.4748, 0.5437, 0.6103, 0.6898, 0.7440, 0.7975, 0.8528, 0.8820] 

# Val Loss dari Epoch 1 sampai 10
val_loss = [3.1703, 2.5801, 2.1613, 1.8792, 1.7682, 1.7039, 1.6798, 1.6697, 1.8352, 1.9079] 

# Val Acc dari Epoch 1 sampai 10
val_acc = [0.3253, 0.4358, 0.5036, 0.5549, 0.5834, 0.6113, 0.6211, 0.6293, 0.6348, 0.6391]

# 2. PROSES PEMBUATAN GRAFIK
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Grafik Kiri: Kurva Loss (Train vs Val)
ax1.plot(epochs, train_loss, 'r-o', label='Training Loss', linewidth=2)
ax1.plot(epochs, val_loss, 'b--s', label='Validation Loss', linewidth=2)
ax1.set_title('Perbandingan Loss (AlexNet pada Caltech-101)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_xticks(epochs)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Grafik Kanan: Kurva Akurasi (Train vs Val)
ax2.plot(epochs, train_acc, 'r-o', label='Training Accuracy', linewidth=2)
ax2.plot(epochs, val_acc, 'b--s', label='Validation Accuracy', linewidth=2)
ax2.set_title('Perbandingan Akurasi (AlexNet pada Caltech-101)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Akurasi')
ax2.set_xticks(epochs)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# Merapikan dan Menyimpan Gambar
plt.tight_layout()
plt.savefig('alexnet_curve_graph.png')
print("Grafik berhasil disimpan sebagai 'alexnet_curve_graph.png'!")
plt.show() # Menampilkan gambar secara langsung