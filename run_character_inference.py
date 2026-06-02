import os
import sys
import json
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
import torchvision.models as models
import pygame
import pygame.midi
import threading
import time

# Dosya yolları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAR_MODEL_PATH = os.path.join(BASE_DIR, "character_model.pth")
CHAR_CONFIG_PATH = os.path.join(BASE_DIR, "character_model_config.json")
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.pth")
MIDI_MODEL_PATH = os.path.join(BASE_DIR, "EN_IYI_MODEL.pth")
MIDI_CONFIG_PATH = os.path.join(BASE_DIR, "EN_IYI_MODEL_config.json")
VIDEO_PATH = os.path.join(BASE_DIR, "lotrTrailer.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "lotrTrailer_annotated.mp4")

# Gerekli sınıfları import edelim
from train_facial_recognition import EmotionCNN, emotion_map
from train_midi_model import MidiEmotionLSTM

# Ters eşleştirme sözlüğü (Duygu ismine ulaşmak için)
idx_to_emotion = {v: k for k, v in emotion_map.items()}

# Türkçe duygu etiketleri
tr_emotion_map = {
    'angry': 'Sinirli',
    'disgust': 'Igrenme',
    'fear': 'Korku',
    'happy': 'Mutlu',
    'neutral': 'Huzurlu',
    'sad': 'Uzgun',
    'surprise': 'Saskin'
}

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def map_emotion_to_midi(emotion_idx):
    """
    Duyguları MIDI çeyreklerine (Quadrants) eşleştirir:
    Q1: Neşeli, coşkulu -> Happy (3), Surprise (6)
    Q2: Gergin, sinirli -> Angry (0), Fear (2)
    Q3: Hüzünlü, melankolik -> Disgust (1), Sad (5)
    Q4: Huzurlu, rahatlamış -> Neutral (4)
    """
    if emotion_idx in [3, 6]:
        return 0  # Q1
    elif emotion_idx in [0, 2]:
        return 1  # Q2
    elif emotion_idx in [1, 5]:
        return 2  # Q3
    else:
        return 3  # Q4

class MidiVibeEngine:
    """Arka planda kesintisiz, nota-nota MIDI üreten ve çalan canlı duygu motoru sınıfı."""
    def __init__(self, midi_model_path, config_path, device):
        self.device = device
        self.current_quadrant = 3  # Varsayılan: Q4 (Huzurlu/Sakin)
        self.running = True
        self.midi_out = None
        self.thread = None
        
        # LSTM Model Parametreleri
        embed_dim = 64
        hidden_dim = 256
        num_layers = 2
        dropout = 0.3
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    embed_dim = config.get("embed_dim", embed_dim)
                    hidden_dim = config.get("hidden_dim", hidden_dim)
                    num_layers = config.get("num_layers", num_layers)
                    dropout = config.get("dropout", dropout)
            except Exception as e:
                print(f"[Müzik Motoru] Config yüklenirken hata: {e}")
                
        self.model = MidiEmotionLSTM(
            emotion_classes=4,
            pitch_classes=128,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        try:
            self.model.load_state_dict(torch.load(midi_model_path, map_location=device, weights_only=True))
            self.model.to(device)
            self.model.eval()
            print("[Müzik Motoru] LSTM MIDI Yapay Zeka Beyni başarıyla yüklendi.")
        except Exception as e:
            print(f"[Müzik Motoru] HATA: Müzik modeli yüklenemedi: {e}")
            self.model = None

        # pygame.midi ile ses kartına doğrudan bağlantı kur
        try:
            pygame.midi.init()
            device_id = pygame.midi.get_default_output_id()
            if device_id == -1:
                # Varsayılan çıkış bulunamazsa, listedeki ilk çıkış aygıtını bul
                for i in range(pygame.midi.get_count()):
                    info = pygame.midi.get_device_info(i)
                    if info and info[3] == 1: # is_output
                        device_id = i
                        break
                        
            if device_id != -1:
                self.midi_out = pygame.midi.Output(device_id)
                # Enstrüman atamaları:
                # Kanal 0: Akustik Gitar (Program 24)
                # Kanal 1: Elektrik Bas Gitar (Program 33)
                self.midi_out.set_instrument(24, channel=0)
                self.midi_out.set_instrument(33, channel=1)
                print(f"[Müzik Motoru] Canlı Ses Aygıtı Başlatıldı: ID {device_id} (Microsoft GS Synth)")
            else:
                print("[Müzik Motoru] UYARI: Varsayılan MIDI çıkış aygıtı bulunamadı.")
        except Exception as e:
            print(f"[Müzik Motoru] MIDI aygıtı başlatılırken hata: {e}")

    def start(self):
        if self.model is None or self.midi_out is None:
            print("[Müzik Motoru] Oynatıcı başlatılamadı (Model veya ses çıkışı eksik).")
            return
        self.thread = threading.Thread(target=self._play_loop)
        self.thread.daemon = True
        self.thread.start()
        print("[Müzik Motoru] Canlı besteleyici iş parçacığı başlatıldı.")

    def update_emotion(self, midi_quadrant):
        if self.current_quadrant != midi_quadrant:
            quadrant_names = {0: "Q1 - Mutlu/Coşkulu", 1: "Q2 - Gergin/Sinirli", 2: "Q3 - Hüzünlü/Melankolik", 3: "Q4 - Huzurlu/Sakin"}
            print(f"[Duygu Motoru] Dinamik Vibe Değişimi -> {quadrant_names[midi_quadrant]}")
            self.current_quadrant = midi_quadrant

    def _play_loop(self):
        # Modelin beslenmesi için son 32 notanın kaydı (Pitch, Velocity, Duration)
        # Orta C (60) tonundan başlayarak dolduruyoruz
        history = [[60, 80, 0.5]] * 32
        
        while self.running:
            quadrant = self.current_quadrant
            emotion_tensor = torch.tensor([quadrant], dtype=torch.long).to(self.device)
            
            # Son 32 notayı model girdisi formatına getir (Batch, Seq_Len, Features)
            notes_tensor = torch.tensor([history[-32:]], dtype=torch.float32).to(self.device)
            
            # Yapay zeka ile sonraki notayı tahmin et
            with torch.no_grad():
                pitch_logits, vel_preds, dur_preds = self.model(notes_tensor, emotion_tensor)
                
                next_pitch_logits = pitch_logits[0, -1, :] 
                next_vel = vel_preds[0, -1].item()
                next_dur = dur_preds[0, -1].item()
                
                # Olasılıksal örnekleme ile çeşitlilik kazandır
                probs = torch.softmax(next_pitch_logits, dim=-1)
                next_pitch = torch.multinomial(probs, 1).item()
                
                # Üst üste çok fazla aynı nota basılmasını engelle
                if len(history) >= 2 and history[-1][0] == next_pitch and history[-2][0] == next_pitch:
                    _, top_class = torch.topk(probs, 3)
                    for p_class in top_class:
                        if p_class.item() != next_pitch:
                            next_pitch = p_class.item()
                            break
                            
                # Değerleri mantıklı sınırlar içerisine çek
                next_vel = max(60, min(120, int(next_vel)))
                next_dur = max(0.3, min(1.0, next_dur))  # Geçişlerin hızlı olması için süreyi kısıtlıyoruz
                
            # Geçmiş listesini güncelle
            history.append([next_pitch, next_vel, next_dur])
            
            # --- NOTALARI CANLI ÇAL ---
            # Melodi (Kanal 0)
            self.midi_out.note_on(next_pitch, next_vel, channel=0)
            
            # Eşlik bas sesi (Kanal 1) - Melodinin 2 oktav altını bas olarak ekle
            if len(history) % 2 == 0:
                bas_pitch = max(36, min(55, next_pitch - 24))
                self.midi_out.note_on(bas_pitch, int(next_vel * 0.85), channel=1)
            else:
                bas_pitch = None
                
            # Nota vuruş süresi kadar bekle
            time.sleep(next_dur)
            
            # Notaları kapat (Sustain olmaması için)
            self.midi_out.note_off(next_pitch, 0, channel=0)
            if bas_pitch is not None:
                self.midi_out.note_off(bas_pitch, 0, channel=1)

    def stop(self):
        self.running = False
        print("[Müzik Motoru] Kapatılıyor, tüm MIDI sesleri kesiliyor...")
        if self.midi_out:
            # Çalan tüm sesleri kapat (Sustur)
            for channel in range(16):
                for pitch in range(128):
                    try:
                        self.midi_out.note_off(pitch, 0, channel)
                    except:
                        pass
            self.midi_out.close()
        pygame.midi.quit()

def main():
    print("=======================================================================")
    print("VIBERYTHM: GERÇEK ZAMANLI KARAKTER TANIMA VE KESİNTİSİZ DUYGU MOTORU")
    print("=======================================================================")
    
    device = get_device()
    print(f"Cihaz: {device}")
    
    # 1. Dosya kontrolleri
    if not os.path.exists(CHAR_MODEL_PATH) or not os.path.exists(CHAR_CONFIG_PATH):
        print("[HATA] Karakter tanıma modeli veya config dosyası bulunamadı!")
        print("Lütfen önce 'augment_characters.py' ve 'train_character_recognition.py' çalıştırın.")
        return
        
    if not os.path.exists(EMOTION_MODEL_PATH):
        print(f"[HATA] Duygu analiz modeli bulunamadı: '{EMOTION_MODEL_PATH}'")
        return
        
    if not os.path.exists(MIDI_MODEL_PATH):
        print(f"[HATA] MIDI Müzik modeli bulunamadı: '{MIDI_MODEL_PATH}'")
        return
        
    if not os.path.exists(VIDEO_PATH):
        print(f"[HATA] Giriş videosu bulunamadı: '{VIDEO_PATH}'")
        return
        
    # 2. Config ve Sınıfları Yükle
    with open(CHAR_CONFIG_PATH, "r", encoding="utf-8") as f:
        char_config = json.load(f)
    char_classes = char_config["classes"]
    num_char_classes = char_config["num_classes"]
    print(f"Tanınacak Karakterler: {char_classes}")
    
    # 3. Karakter Tanıma Modelini Yükle (ResNet-18)
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        char_model = resnet18(weights=None)
    except ImportError:
        char_model = models.resnet18(pretrained=False)
    char_model.fc = nn.Linear(char_model.fc.in_features, num_char_classes)
    char_model.load_state_dict(torch.load(CHAR_MODEL_PATH, map_location=device, weights_only=True))
    char_model.to(device)
    char_model.eval()
    print("[+] Karakter tanıma modeli yüklendi.")
    
    # 4. Duygu Analiz Modelini Yükle (EmotionCNN)
    emotion_model = EmotionCNN(num_classes=7)
    emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=device, weights_only=True))
    emotion_model.to(device)
    emotion_model.eval()
    print("[+] Duygu tanıma modeli yüklendi.")
    
    # Pygame Başlat
    pygame.init()
    
    # Müzik motorunu başlat ve arka plan çalıcısını çalıştır
    vibe_engine = MidiVibeEngine(MIDI_MODEL_PATH, MIDI_CONFIG_PATH, device)
    vibe_engine.start()
    
    # 5. MediaPipe Yüz Algılayıcı Kurulumu
    from mediapipe.tasks import python
    from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
    
    tflite_model_path = os.path.join(BASE_DIR, 'blaze_face_short_range.tflite')
    base_options = python.BaseOptions(model_asset_path=tflite_model_path)
    options = FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
    
    # Ön işleme transformları
    char_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    emotion_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Video okuyucu ve yazıcı kurulumu
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    
    show_gui = True
    frame_count = 0
    char_threshold = 0.70  # Karakter tanıma olasılık eşiği
    
    print("\n--- Analiz Başladı ---")
    print("Çıkmak için ekrandaki pencereye tıklayıp klavyeden 'q' tuşuna basın.")
    
    with FaceDetector.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            results = detector.detect(mp_image)
            frame_emotions = []
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.bounding_box
                    x = int(bbox.origin_x)
                    y = int(bbox.origin_y)
                    w_box = int(bbox.width)
                    h_box = int(bbox.height)
                    
                    # Sınır kontrolü
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(width, x + w_box)
                    y2 = min(height, y + h_box)
                    
                    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                        continue
                        
                    # Yüz alanlarını kırp
                    face_crop_rgb = rgb_frame[y1:y2, x1:x2]
                    face_crop_gray = gray_frame[y1:y2, x1:x2]
                    
                    # --- 1. AŞAMA: Karakter Kimlik Teşhisi ---
                    char_tensor = char_transform(face_crop_rgb).unsqueeze(0).to(device)
                    with torch.no_grad():
                        char_outputs = char_model(char_tensor)
                        char_probs = torch.softmax(char_outputs, dim=1)[0]
                        max_char_prob, pred_char_idx = torch.max(char_probs, 0)
                        
                    if max_char_prob.item() >= char_threshold:
                        char_name = char_classes[pred_char_idx].upper()
                    else:
                        char_name = "BILINMEYEN"
                        
                    # --- 2. AŞAMA: Karakter Duygu Teşhisi ---
                    emotion_tensor = emotion_transform(face_crop_gray).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emotion_outputs = emotion_model(emotion_tensor)
                        emotion_probs = torch.softmax(emotion_outputs, dim=1)[0]
                        max_emo_prob, pred_emo_idx = torch.max(emotion_probs, 0)
                        
                    emotion_label = idx_to_emotion.get(pred_emo_idx.item(), "neutral")
                    tr_emotion = tr_emotion_map.get(emotion_label, "Huzurlu")
                    
                    # Karedeki duyguları listeye ekle
                    frame_emotions.append(pred_emo_idx.item())
                    
                    # --- 3. AŞAMA: Çizim İşlemleri ---
                    # Bounding Box çiz
                    color = (0, 255, 0) if char_name != "BILINMEYEN" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Karakter ismi ve duyguyu yaz (Örn: FRODO - MUTLU)
                    label_text = f"{char_name} | {tr_emotion.upper()}"
                    cv2.putText(frame, label_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            
            # --- 4. AŞAMA: Duygu Motoru Güncelleme ---
            # Sahnedeki baskın duyguya göre müziğin "vibe"ını güncelle (kesintisiz geçiş yapar)
            if frame_emotions:
                dominant_emotion = max(set(frame_emotions), key=frame_emotions.count)
                midi_quadrant = map_emotion_to_midi(dominant_emotion)
                vibe_engine.update_emotion(midi_quadrant)
                
            # İşlenmiş kareyi kaydet
            out.write(frame)
            
            # GUI Ekranı göster
            if show_gui:
                try:
                    cv2.imshow('VibeRythm: Karakter Tanima ve Duygu Motoru', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("İşlem durduruldu.")
                        break
                except cv2.error:
                    show_gui = False
                    print("[UYARI] GUI ortamı bulunamadı, video arka planda kaydediliyor...")
                    
            if frame_count % 50 == 0 or frame_count == total_frames:
                print(f"İlerleme: {frame_count}/{total_frames} (%{100*frame_count/total_frames:.1f})")
                
    # Temizlik işlemleri
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Müzik motorunu kapat
    vibe_engine.stop()
    pygame.quit()
    
    print(f"\nİşlem tamamlandı! Analiz edilen video kaydedildi: '{OUTPUT_VIDEO_PATH}'")

if __name__ == "__main__":
    main()
