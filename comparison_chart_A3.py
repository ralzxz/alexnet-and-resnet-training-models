import matplotlib.pyplot as plt
import os

# =========================================================================
# DATA TEREKAM DARI TERMINAL
# =========================================================================

# 1. Data dari Model ALEXNET LAMA (Sebelum perbaikan)
base_train_acc = [0.2060, 0.3706, 0.4748, 0.5437, 0.6103, 0.6898, 0.7440, 0.7975, 0.8528, 0.8820]
base_val_acc   = [0.3253, 0.4358, 0.5036, 0.5549, 0.5834, 0.6113, 0.6211, 0.6293, 0.6348, 0.6391]

# 2. Data dari Model ALEXNET IMPROVED (Setelah pakai Augmentasi + Scheduler)
imp_train_acc  = [0.1904, 0.3341, 0.4130, 0.4776, 0.5662, 0.5892, 0.5952, 0.6063, 0.6309, 0.6231]
imp_val_acc    = [0.2974, 0.3723, 0.4303, 0.5074, 0.5429, 0.5451, 0.5528, 0.5610, 0.5653, 0.5637]

# =========================================================================

epochs = range(1, 11)

# Bikin 1 kotak grafik berukuran proporsional
plt.figure(figsize=(10, 6))

# --- PLOT 4 GARIS DI SATU KOTAK ---
# 1. Model Lama (Garis Putus-putus)
plt.plot(epochs, base_train_acc, color='blue', linestyle='--', alpha=0.6, label='Train Acc (Base AlexNet)')
plt.plot(epochs, base_val_acc, color='red', linestyle='--', alpha=0.6, label='Val Acc (Base AlexNet)')

# 2. Model Baru (Garis Tebal dengan Titik)
plt.plot(epochs, imp_train_acc, color='blue', marker='o', linewidth=2, label='Train Acc (Improved AlexNet)')
plt.plot(epochs, imp_val_acc, color='red', marker='o', linewidth=2, label='Val Acc (Improved AlexNet)')

# Dekorasi Grafik
plt.title('Perbandingan Akurasi: Base vs Improved AlexNet', fontsize=16, pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Akurasi', fontsize=12)

# Taruh legenda di sudut kanan bawah biar nggak menutupi garis
plt.legend(loc='lower right', fontsize=10) 
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()

# Simpan otomatis ke folder Images
save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Images')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'comparison_chart_A3.png')

plt.savefig(save_path, dpi=300) 
print(f"\nSelesai! Grafik perbandingan A3 berhasil disimpan di:\n{save_path}")

plt.show()