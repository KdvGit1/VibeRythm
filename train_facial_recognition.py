import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Dosya yolları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "FaceRecognitionData", "images")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "validation")

# 1. Emotion (Duygu) Map'ini oluştur
# Sabit (hardcoded) bir liste olarak tanımlıyoruz
emotion_map = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}

def prepare_data(directory):
    """
    Belirtilen dizindeki (train veya validation) fotoğraf yollarını ve 
    emotion map'teki sayısal karşılıklarını (label) listeler halinde döner.
    """
    filepaths = []
    labels = []
    
    for emotion, label in emotion_map.items():
        emotion_dir = os.path.join(directory, emotion)
        if not os.path.isdir(emotion_dir):
            continue
            
        # O duygu sınıfındaki tüm resimlerin yollarını listele
        valid_extensions = ('.jpg', '.jpeg', '.png')
        for img_name in os.listdir(emotion_dir):
            if img_name.lower().endswith(valid_extensions):
                img_path = os.path.join(emotion_dir, img_name)
                filepaths.append(img_path)
                labels.append(label)
            
    # Modeli eğitirken sıranın öğrenmeyi ezberletmemesi için verileri karıştır (shuffle)
    combined = list(zip(filepaths, labels))
    random.shuffle(combined)
    
    # Her bir elemanı (fotoğraf_yolu, etiket) tuple olan tek bir liste döndür
    return combined

# --- YAPAY ZEKA (AI) EĞİTİM MODÜLÜ ---

def get_device():
    """Hangi donanımın (GPU/CPU) müsait olduğunu tespit eder."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class EmotionDataset(Dataset):
    """PyTorch için özel veri seti okuyucu sınıfı."""
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        
        # Resmi siyah beyaz (Grayscale) açıyoruz
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def create_dataloaders(train_data, val_data, batch_size=64):
    """Verileri PyTorch DataLoader formatına çevirerek batch grupları oluşturur."""
    
    # Train için Veri Çeşitlendirme (Data Augmentation) - Model daha zor ezberlesin
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Hafif döndürme toleransı
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Validation için sadece normalleştirme
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = EmotionDataset(train_data, transform=train_transform)
    val_dataset = EmotionDataset(val_data, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class EmotionCNN(nn.Module):
    """Daha Geniş Çaplı (Deeper Mini-VGG) Custom CNN Modeli."""
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 48x48 -> 24x24
            nn.Dropout2d(0.2),  # Ezberleme toleransı
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 24x24 -> 12x12
            nn.Dropout2d(0.2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 12x12 -> 6x6
            nn.Dropout2d(0.2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 6x6 -> 3x3
            nn.Dropout2d(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Modelin bir tur (epoch) eğitim sürecini gerçekleştirir."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return (running_loss / total), (100. * correct / total)

def validate_epoch(model, dataloader, criterion, device):
    """Modelin daha önce görmediği validation testinde başarımını ölçer."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return (running_loss / total), (100. * correct / total)

def run_training_pipeline(train_data, val_data, epochs=5):
    """Baştan sona modelin eğitim döngüsünü kontrol eder."""
    print("\n--- Yapay Zeka Eğitimi (AI Training) Başlıyor ---")
    
    device = get_device()
    print(f"Kullanılacak Donanım: {device}")
    
    train_loader, val_loader = create_dataloaders(train_data, val_data, batch_size=64)
    model = EmotionCNN(num_classes=7).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: %{train_acc:.2f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: %{val_acc:.2f}")
              
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "emotion_model.pth")
            print("  --> En iyi model gelişti! 'emotion_model.pth' kaydedildi.")

if __name__ == "__main__":
    print("Fotoğraflar emotion map ile eşleştiriliyor...")
    base_train = prepare_data(TRAIN_DIR)
    base_val = prepare_data(VAL_DIR)
    
    print(f"Eğitim (Train) seti: {len(base_train)} fotoğraf hazır.")
    print(f"Doğrulama (Validation) seti: {len(base_val)} fotoğraf hazır.")
    
    # Eğitimi başlat (Epoch 30 olarak arttırıldı)
    run_training_pipeline(base_train, base_val, epochs=30)