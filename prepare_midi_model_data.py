import os
import torch
import pandas as pd
import numpy as np
import pretty_midi
import random
from torch.utils.data import Dataset, DataLoader

# 1. Veri Yolu Tanımları
BASE_DIR = r"C:\Users\KDV\Desktop\VibeRythm"
MIDI_DATA_DIR = os.path.join(BASE_DIR, "MidiData")
MIDI_DIR = os.path.join(MIDI_DATA_DIR, "midis")
LABEL_CSV = os.path.join(MIDI_DATA_DIR, "label.csv")

TRAIN_DATA_PATH = os.path.join(MIDI_DATA_DIR, "train_ready_data.pt")
TEST_DATA_PATH = os.path.join(MIDI_DATA_DIR, "test_ready_data.pt")

"""
Q1 — Neşeli, coşkulu (Midi Q1 -> Happy, Surprise)
Q2 — Gergin, sinirli (Midi Q2 -> Angry, Fear)
Q3 — Hüzünlü, melankolik (Midi Q3 -> Disgust, Sad)
Q4 — Huzurlu, rahatlamış (Midi Q4 -> Neutral)
"""

# Duyguları Çeyreklere (Quadrants) Eşleştirme
midi_map = {
    "Q1": [3, 6],   # Happy, Surprise
    "Q2": [0, 2],   # Angry, Fear
    "Q3": [1, 5],   # Disgust, Sad
    "Q4": [4]       # Neutral
}

# --- 1. VERİ İŞLEME VE KAYDETME FONKSİYONU ---
def prepare_datasets(csv_file, midi_dir, out_dir, test_size=0.3):
    """
    Tüm MIDI dosyalarını parse eder, her şarkının notalarını 
    tensör olarak ayrıştırır ve %70 Eğitim (Train) - %30 Test olarak böler/kaydeder.
    """
    print("Veriler önceden işleniyor. Binlerce MIDI dosyasını okumak 1-2 dakika sürebilir...")
    data_frame = pd.read_csv(csv_file)
    dataset_list = []
    
    for idx in range(len(data_frame)):
        midi_name = str(data_frame.iloc[idx, 0]) + ".mid"
        midi_path = os.path.join(midi_dir, midi_name)
        emotion_label = int(data_frame.iloc[idx, 1]) - 1 # (1,2,3,4 değerini loss algılasın diye 0,1,2,3'e çekiyoruz)
        
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            notes = []
            
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        pitch = note.pitch
                        
                        # 3. Kere aynı nota gelmesini engelle (Tekrar Eden Nota Filtresi)
                        if len(notes) >= 2 and notes[-1][0] == pitch and notes[-2][0] == pitch:
                            continue # Ardışık aynı 3. nota ise görmezden gel ve döngüden atla
                            
                        duration = note.end - note.start
                        notes.append([pitch, note.velocity, duration])
            
            # İçinde nota olan enstrüman kayıtlarını listeye atıyoruz
            if len(notes) > 0:
                notes_tensor = torch.tensor(notes, dtype=torch.float32)
                dataset_list.append({
                    "notes": notes_tensor, 
                    "label": emotion_label,
                    "id": midi_name
                })
        except Exception as e:
            # Uyumsuz/bozuk bir MIDI dosyası atla
            pass
            
    print(f"Toplam başarıyla parse edilen şarkı sayısı: {len(dataset_list)}")
    
    # Kayıtları karıştırıyoruz
    random.shuffle(dataset_list)
    
    # Array'i len(list)*0.3 kadar bölüyoruz
    split_idx = int(len(dataset_list) * (1 - test_size))
    train_data = dataset_list[:split_idx]
    test_data = dataset_list[split_idx:]
    
    train_path = os.path.join(out_dir, "train_ready_data.pt")
    test_path = os.path.join(out_dir, "test_ready_data.pt")
    
    # PT datalarını kaydediyoruz (PyTorch formatı)
    torch.save(train_data, train_path)
    torch.save(test_data, test_path)
    
    print(f"\nYÜKLEME BAŞARILI!")
    print(f"--> EĞİTİM DOSYASI:\n   İçerik: {len(train_data)} Şarkı\n   Konum: '{train_path}'")
    print(f"--> TEST DOSYASI:\n   İçerik: {len(test_data)} Şarkı\n   Konum: '{test_path}'\n")

# --- 2. HIZLI TENSOR OKUYUCU NEW DATASET CLASS'I ---
class MidiEmotionDataset(Dataset):
    def __init__(self, pt_data_path, seq_length=50):
        """
        Argümanlar:
            pt_data_path (string): Hazırlanmış train_ready_data.pt veya test_ready_data.pt yolu.
            seq_length (int): RNN/LSTM'e verilecek ardışık nota sınırı.
        """
        # Hızlı Data Loader İçin Diske kaydettiğimiz tek bir PT array'ini okuyoruz
        # Artık loop içinde binlerce kez pretty_midi objesi oluşturmayacağız
        self.data_list = torch.load(pt_data_path, weights_only=False)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        notes_tensor = item["notes"]
        emotion_label = item["label"]
        
        num_notes = notes_tensor.size(0)
        
        # O şarkıda ne kadar ardışık nota var?
        if num_notes >= self.seq_length:
            # Bol notalıysa rastgele bir yerden 'seq_length' kadar kısmı kes al
            start_idx = np.random.randint(0, num_notes - self.seq_length) if num_notes > self.seq_length else 0
            notes_array = notes_tensor[start_idx : start_idx + self.seq_length]
        else:
            # Fakir notalıysa, boşlukları [0, 0, 0.0] tensoru ile doldur (Padding)
            padding_size = self.seq_length - num_notes
            padding = torch.zeros((padding_size, 3), dtype=torch.float32)
            notes_array = torch.cat((notes_tensor, padding), dim=0)
            
        emotion_tensor = torch.tensor(emotion_label, dtype=torch.long)

        return notes_array, emotion_tensor

# --- KULLANIM SİSTEMATİĞİ ---
if __name__ == "__main__":
    
    # 0) Eğitim verisi yoksa bir kereliğine MIDI klasörünü okuyan veriyi dönüştürsün:
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        prepare_datasets(csv_file=LABEL_CSV, midi_dir=MIDI_DIR, out_dir=MIDI_DATA_DIR, test_size=0.3)
    
    print("[BILGI] Veriseti Diskten Yükleniyor...")
    
    # 1) Çok hızlıca PT dosyalarını model bellek aktarımı modunda çekiyoruz
    train_dataset = MidiEmotionDataset(pt_data_path=TRAIN_DATA_PATH, seq_length=32)
    test_dataset = MidiEmotionDataset(pt_data_path=TEST_DATA_PATH, seq_length=32)
    
    # 2) DataLoader (Batched Processing için)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Örnek Çıktı Testi
    for batch_notes, batch_emotions in train_loader:
        print(f"Eğitim - Notalar Batch Boyutu: {batch_notes.shape}")    
        print(f"Eğitim - Duygu Label Boyutu: {batch_emotions.shape}")  
        print(f"Örnek Bir Şarkının İlk Notası (Pitch, Vol, Dur): {batch_notes[0][0].tolist()}")
        break