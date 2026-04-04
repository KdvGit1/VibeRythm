import os
import json
import itertools
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from prepare_midi_model_data import (
    MidiEmotionDataset, TRAIN_DATA_PATH, TEST_DATA_PATH, 
    prepare_datasets, LABEL_CSV, MIDI_DIR, MIDI_DATA_DIR, BASE_DIR
)

# 1. Model Mimarisi (MidiEmotionLSTM) - Parametrik Yapı
class MidiEmotionLSTM(nn.Module):
    def __init__(self, emotion_classes=4, pitch_classes=128, embed_dim=64, hidden_dim=256, num_layers=2, dropout=0.3):
        super(MidiEmotionLSTM, self).__init__()
        self.emotion_emb = nn.Embedding(emotion_classes, embed_dim)
        self.pitch_emb = nn.Embedding(pitch_classes, embed_dim)
        input_size = embed_dim * 2 + 2 
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc_pitch = nn.Linear(hidden_dim, pitch_classes)
        self.fc_velocity = nn.Linear(hidden_dim, 1)
        self.fc_duration = nn.Linear(hidden_dim, 1)
        
    def forward(self, notes, emotions):
        pitches = notes[:, :, 0].long()
        velocities = notes[:, :, 1].float() / 127.0
        durations = notes[:, :, 2].float()
        pitch_embedded = self.pitch_emb(pitches)
        emo_emb = self.emotion_emb(emotions)
        emo_emb = emo_emb.unsqueeze(1).expand(-1, pitches.size(1), -1)
        vel_dur = torch.stack([velocities, durations], dim=-1)
        lstm_in = torch.cat([pitch_embedded, emo_emb, vel_dur], dim=-1)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.dropout(lstm_out)
        pitch_logits = self.fc_pitch(lstm_out)
        velocity_preds = self.fc_velocity(lstm_out).squeeze(-1) * 127.0
        duration_preds = self.fc_duration(lstm_out).squeeze(-1)
        return pitch_logits, velocity_preds, duration_preds

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

def train_one_model(config, model_dir, train_loader, val_loader, device):
    """
    Belirli bir konfigürasyona sahip tek bir modeli eğitir, kaydeder ve val_loss'u döndürür.
    """
    print(f"\n[>>>] MODEL EGITILIYOR: {config}")
    
    # Config dosyasını kaydet
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    model = MidiEmotionLSTM(
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    criterion_pitch = nn.CrossEntropyLoss()
    criterion_continuous = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    
    max_epochs = 2000 # Erken durdurma ile kesilecektir (Taramaları hızlandırmak için max 200 epoch yapıldı)
    early_stopper = EarlyStopping(patience=10, min_delta=0.01)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        for batch_notes, batch_emotions in train_loader:
            batch_notes = batch_notes.to(device)
            batch_emotions = batch_emotions.to(device)
            
            x_notes = batch_notes[:, :-1, :]
            y_notes = batch_notes[:, 1:, :] 
            
            optimizer.zero_grad()
            pitch_logits, vel_preds, dur_preds = model(x_notes, batch_emotions)
            
            target_pitch = y_notes[:, :, 0].long()
            target_vel = y_notes[:, :, 1].float()
            target_dur = y_notes[:, :, 2].float()
            
            loss_pitch = criterion_pitch(pitch_logits.transpose(1, 2), target_pitch)
            loss_vel = criterion_continuous(vel_preds, target_vel)
            loss_dur = criterion_continuous(dur_preds, target_dur)
            
            total_loss = loss_pitch + (loss_vel * 0.005) + (loss_dur * 0.1)
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += total_loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_notes, batch_emotions in val_loader:
                batch_notes = batch_notes.to(device)
                batch_emotions = batch_emotions.to(device)
                x_notes = batch_notes[:, :-1, :]
                y_notes = batch_notes[:, 1:, :]
                
                pitch_logits, vel_preds, dur_preds = model(x_notes, batch_emotions)
                target_pitch = y_notes[:, :, 0].long()
                target_vel = y_notes[:, :, 1].float()
                target_dur = y_notes[:, :, 2].float()
                
                v_loss_pitch = criterion_pitch(pitch_logits.transpose(1, 2), target_pitch)
                v_loss_vel = criterion_continuous(vel_preds, target_vel)
                v_loss_dur = criterion_continuous(dur_preds, target_dur)
                
                v_total_loss = v_loss_pitch + (v_loss_vel * 0.005) + (v_loss_dur * 0.1)
                total_val_loss += v_total_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
        print(f"  Epoch [{epoch:03d}/{max_epochs:03d}] - Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")
        
        if early_stopper(avg_val_loss):
            print(f"  [!] Early Stopping. Son {early_stopper.patience} turdur iyilesme yok.")
            break
            
    # Modeli Kaydet
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  [*] Egitim bitti. Model kaydedildi: {model_path} | En Iyi Loss: {best_val_loss:.4f}")
    
    return best_val_loss

def grid_search_auto_tune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=======================================================")
    print(f"[*] Cihaz: {str(device).upper()}")
    print("[*] HIPERPARAMETRE GRID SEARCH (OTOMATIK TARAMA) BASLADI")
    print("=======================================================\n")
    
    # 1. Veriseti Hazırlıkları
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        print(f"[*] Egitim verisi bulanamadı ({TRAIN_DATA_PATH}). Ilk islem baslatiliyor...")
        prepare_datasets(LABEL_CSV, MIDI_DIR, MIDI_DATA_DIR)
        
    print("[*] Veriseti Diskten RAM'e aliniyor...")
    train_dataset = MidiEmotionDataset(pt_data_path=TRAIN_DATA_PATH, seq_length=32)
    val_dataset = MidiEmotionDataset(pt_data_path=TEST_DATA_PATH, seq_length=32)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 2. Hyperparameter Grid (Tarama Uzayı)
    param_grid = {
        'embed_dim': [32, 64, 128],
        'hidden_dim': [128, 256, 512],
        'num_layers': [2, 3, 4],
        'dropout': [0.3, 0.5],
        'lr': [0.001]
    }
    
    # Varyasyon kombinasyonlarını çıkart
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"[*] Toplam {len(combinations)} farkli konfigürasyon (Zeka Varyasyonu) olusturuldu.")
    
    AUTOTUNE_DIR = os.path.join(BASE_DIR, "AutoTuneModels")
    os.makedirs(AUTOTUNE_DIR, exist_ok=True)
    
    # Tüm modellerin performans kayıtları
    model_results = []
    
    # 3. Yıkıcı Eğitim Döngüsü (Tüm Kombinasyonları Vurur)
    for idx, combo in enumerate(combinations, start=1):
        # Combo bir tuple'dır, dict'e çevirelim
        config = dict(zip(keys, combo))
        
        # O modele özel klasör
        model_dir = os.path.join(AUTOTUNE_DIR, f"Model_{idx}")
        
        # Eğitimi başlat
        best_val_loss = train_one_model(config, model_dir, train_loader, val_loader, device)
        
        # Sonuçları Kaydet
        model_results.append({
            "model_id": idx,
            "folder": model_dir,
            "config": config,
            "val_loss": best_val_loss
        })
        
    # 4. SKOR TABLOSU (LEADERBOARD) OLUŞTURMA
    print("\n\n=======================================================")
    print("====== SONUC BILDIRGESI (LEADERBOARD) ===================")
    print("=======================================================\n")
    
    # En iyi (en düşük loss) olan modele göre sırala
    sorted_results = sorted(model_results, key=lambda x: x["val_loss"])
    
    for i, res in enumerate(sorted_results, start=1):
        mid = res["model_id"]
        vloss = res["val_loss"]
        conf = res["config"]
        print(f"{i}. Model_{mid} | Loss: {vloss:.4f} | Config: E={conf['embed_dim']}, H={conf['hidden_dim']}, L={conf['num_layers']}, D={conf['dropout']}")
        
    # 5. EN İYİ MODELİ KOPYALAYIP TAÇLANDIRMA
    best_candidate = sorted_results[0]
    best_original_path = os.path.join(best_candidate["folder"], "model.pth")
    target_path = os.path.join(BASE_DIR, "EN_IYI_MODEL.pth")
    
    # Kopyalama işlemi (En iyi modeli ana dizine cikarir)
    shutil.copyfile(best_original_path, target_path)
    
    best_config_path = os.path.join(best_candidate["folder"], "config.json")
    target_config_path = os.path.join(BASE_DIR, "EN_IYI_MODEL_config.json")
    if os.path.exists(best_config_path):
        shutil.copyfile(best_config_path, target_config_path)
    
    print(f"\n[***] TACI ALAN SIZINTI: Model_{best_candidate['model_id']} en dusuk hata payini ({best_candidate['val_loss']:.4f}) elde etti!")
    print(f"Yapay zeka beyni ana dizine kopyalandi -> {target_path}")
    print("[***] Egitim Modulu Kapaniyor, inferans kodunda bu yeni modeli cagirabilirsiniz!")

if __name__ == "__main__":
    grid_search_auto_tune()
