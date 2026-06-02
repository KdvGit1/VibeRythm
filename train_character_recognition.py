import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.models as models

# Dosya yolları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUG_DIR = os.path.join(BASE_DIR, "lotr_characters", "augmented")
MODEL_PATH = os.path.join(BASE_DIR, "character_model.pth")
CONFIG_PATH = os.path.join(BASE_DIR, "character_model_config.json")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    print("\n--- Model Eğitimi Başlıyor ---")
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
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
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: %{epoch_acc:.2f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: %{epoch_val_acc:.2f}")
              
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("  --> En iyi model gelişti! 'character_model.pth' kaydedildi.")
            
    print(f"\nEğitim tamamlandı! En iyi Validation Doğruluğu: %{best_acc:.2f}")

def main():
    if not os.path.exists(AUG_DIR) or len(os.listdir(AUG_DIR)) == 0:
        print("[HATA] 'lotr_characters/augmented' dizini boş veya bulunamadı!")
        print("Lütfen önce 'python augment_characters.py' komutunu çalıştırarak veri çoğaltma yapın.")
        return
        
    device = get_device()
    print(f"Kullanılacak donanım: {device}")
    
    # Standart ImageNet normalizasyonu ile veri yükleme kuralları
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ImageFolder ile sınıfları dizin adlarından otomatik oku
    dataset = datasets.ImageFolder(root=AUG_DIR, transform=transform)
    classes = dataset.classes
    num_classes = len(classes)
    
    if num_classes < 1:
        print("[HATA] Hiçbir karakter sınıfı bulunamadı. Lütfen input klasörüne resim yükleyin.")
        return
        
    print(f"Tespit edilen karakter sınıfları ({num_classes} adet): {classes}")
    
    # Sınıf eşleştirmesini JSON olarak kaydet
    config = {
        "classes": classes,
        "num_classes": num_classes
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"Sınıf eşleştirmeleri '{CONFIG_PATH}' dosyasına kaydedildi.")
    
    # Veri kümesini %80 Eğitim, %20 Doğrulama olarak böl
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Pre-trained ResNet-18 modelini yükle (Geriye dönük uyumluluk kontrollü)
    print("Pre-trained ResNet-18 modeli yükleniyor...")
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except ImportError:
        model = models.resnet18(pretrained=True)
        
    # Son katmanı (fc) karakter sayımıza uyacak şekilde değiştir
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Hızlı ve etkili transfer öğrenmesi için optimizasyon
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)

if __name__ == "__main__":
    main()
