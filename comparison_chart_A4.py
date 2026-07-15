import matplotlib.pyplot as plt
import os

# =========================================================================
# 1. DATA IMPROVED ALEXNET (Dari A3)
# =========================================================================
alex_train_acc  = [0.1904, 0.3341, 0.4130, 0.4776, 0.5662, 0.5892, 0.5952, 0.6063, 0.6309, 0.6231]
alex_val_acc    = [0.2974, 0.3723, 0.4303, 0.5074, 0.5429, 0.5451, 0.5528, 0.5610, 0.5653, 0.5637]

# =========================================================================
# 2. DATA CUSTOM RESNET (Dari Terminal Aziz)
# =========================================================================
resnet_train_acc = [0.2555, 0.3213, 0.3621, 0.3880, 0.4123, 0.4346, 0.4521, 0.4759, 0.4947, 0.5083]
resnet_val_acc   = [0.3029, 0.3428, 0.3778, 0.3827, 0.4019, 0.4467, 0.4609, 0.4707, 0.4872, 0.4932]

epochs = range(1, 11)

plt.figure(figsize=(10, 6))

# Plot Improved AlexNet (Garis Putus-putus)
plt.plot(epochs, alex_train_acc, color='blue', linestyle='--', alpha=0.6, label='Train Acc (Improved AlexNet)')
plt.plot(epochs, alex_val_acc, color='red', linestyle='--', alpha=0.6, label='Val Acc (Improved AlexNet)')

# Plot Custom ResNet (Garis Tebal + Titik)
plt.plot(epochs, resnet_train_acc, color='blue', marker='o', linewidth=2, label='Train Acc (Custom ResNet)')
plt.plot(epochs, resnet_val_acc, color='red', marker='o', linewidth=2, label='Val Acc (Custom ResNet)')

plt.title('Perbandingan Performa: Improved AlexNet vs Custom ResNet', fontsize=16, pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Akurasi', fontsize=12)

plt.legend(loc='lower right', fontsize=10) 
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Images')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'resnet_vs_alexnet_chart_A4.png')
plt.savefig(save_path, dpi=300) 

print(f"\nSelesai! Grafik A4 berhasil disimpan di:\n{save_path}")
plt.show()