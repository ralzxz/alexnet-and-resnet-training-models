import os
import sys
import pygame
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Konfigurasi
MODEL_PATH = "mnist_lenet.pth"
EPOCHS = 5 # Mengubah epoch jadi 5 karena ini contoh aplikasi dasar
BATCH_SIZE = 128
LR = 1e-3
BRUSH_SIZE = 14
CANVAS_SIZE = 280
NORM_MEAN = (0.1307,)
NORM_STD = (0.3081,)

# Definisi Model
class LeNetMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Transformasi Data
def get_train_transform():
    return transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

def get_infer_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

# Proses Pelatihan
def train_model(device):
    print(f"[Train] Melatih pada: {device}")
    train_ds = datasets.MNIST(root="./mnist", train=True, download=True, transform=get_train_transform())
    val_ds = datasets.MNIST(root="./mnist", train=False, download=True, transform=get_infer_transform())
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    
    model = LeNetMNIST().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

        # Validasi
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                correct += model(data).argmax(1).eq(target).sum().item()
                total += data.size(0)
        
        val_acc = correct / total
        print(f"Epoch {epoch}/{EPOCHS} | Akurasi Val={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    # Kembalikan model ke CPU untuk penggunaan GUI
    cpu_model = LeNetMNIST()
    # weights_only=True ditambahkan untuk keamanan file di pytorch terbaru
    cpu_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    cpu_model.eval()
    return cpu_model

# Muat Model (Deteksi Otomatis)
def load_model():
    model = LeNetMNIST()
    if not os.path.exists(MODEL_PATH):
        # Jika file model tidak ada, lakukan pelatihan otomatis
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return train_model(device)
    
    # Jika file ada, langsung muat bobotnya
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model

# Pra-pemrosesan Kanvas
def preprocess_canvas(surface):
    raw = pygame.surfarray.array3d(surface) # (W,H,3)
    img = Image.fromarray(raw.transpose(1, 0, 2)) # (H,W,3)
    img = img.convert("L")
    img = ImageOps.invert(img)
    
    bbox = img.getbbox()
    if bbox is None:
        blank = Image.new("L", (28, 28), 0)
        return get_infer_transform()(blank).unsqueeze(0)
        
    img = img.crop(bbox)
    max_side = max(img.size)
    margin = max_side // 4
    new_size = max_side + 2 * margin
    
    padded = Image.new("L", (new_size, new_size), 0)
    offset = ((new_size - img.size[0]) // 2, (new_size - img.size[1]) // 2)
    padded.paste(img, offset)
    
    img_28 = padded.resize((28, 28), Image.LANCZOS)
    return get_infer_transform()(img_28).unsqueeze(0)

# Kelas Tombol GUI
class Button:
    def __init__(self, label, x, y, w, h, color):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.color = color

    def draw(self, screen, font):
        pygame.draw.rect(screen, self.color, self.rect, border_radius=6)
        screen.blit(font.render(self.label, True, (0, 0, 0)), (self.rect.x + 10, self.rect.y + 8))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# Loop Utama GUI
def run_gui(model):
    pygame.init()
    screen = pygame.display.set_mode((450, 280))
    pygame.display.set_caption("Klasifikasi Digit MNIST")
    ticker = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    
    canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
    canvas.fill((255, 255, 255))
    
    classify_btn = Button("Klasifikasi", 295, 20, 140, 38, (50, 200, 80))
    clear_btn = Button("Hapus", 295, 68, 140, 38, (220, 60, 60))
    
    pred_text = "Gambar angka ->"
    conf_text = ""
    drawing = False
    last_pos = None

    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if classify_btn.is_clicked(mouse_pos):
                    x = preprocess_canvas(canvas)
                    with torch.no_grad():
                        probs = torch.softmax(model(x), dim=1)
                        pred = probs.argmax(1).item()
                        conf = probs.max().item()
                        pred_text = f"Prediksi: {pred}"
                        conf_text = f"Keyakinan: {conf*100:.1f}%"
                elif clear_btn.is_clicked(mouse_pos):
                    canvas.fill((255, 255, 255))
                    pred_text = "Gambar angka ->"
                    conf_text = ""
                elif mouse_pos[0] < CANVAS_SIZE:
                    drawing = True
                    last_pos = mouse_pos
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.MOUSEMOTION and drawing:
                if last_pos and last_pos[0] < CANVAS_SIZE:
                    pygame.draw.line(canvas, (0, 0, 0), last_pos, mouse_pos, BRUSH_SIZE)
                    last_pos = mouse_pos

        screen.fill((45, 45, 45))
        screen.blit(canvas, (0, 0))
        pygame.draw.rect(screen, (120, 120, 120), (0, 0, CANVAS_SIZE, CANVAS_SIZE), 2)
        
        screen.blit(font.render(pred_text, True, (255, 220, 50)), (295, 120))
        screen.blit(font.render(conf_text, True, (180, 220, 255)), (295, 150))
        
        classify_btn.draw(screen, font)
        clear_btn.draw(screen, font)
        
        pygame.display.flip()
        ticker.tick(60)

if __name__ == "__main__":
    model = load_model()
    run_gui(model)