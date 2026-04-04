import cv2
import torch
import mediapipe as mp
from torchvision import transforms
from train_facial_recognition import EmotionCNN, get_device, emotion_map

# Ters eşleştirme sözlüğü (Sayı endeksinden -> Duygu ismine ulaşmak için)
idx_to_emotion = {v: k for k, v in emotion_map.items()}

def main():
    print("Orijinal modelin ağırlıkları 'emotion_model.pth' bekleniyor...")
    device = get_device()
    
    # Modeli Yükle
    model = EmotionCNN(num_classes=7)
    try:
        model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
    except FileNotFoundError:
        print("HATA: 'emotion_model.pth' dosyası bulunamadı.")
        print("Lütfen önce 'python train_facial_recognition.py' çalıştırıp modelin eğitilmesini bitirin.")
        return
        
    model.to(device)
    model.eval() # Inference (Test) modu - Ağırlıkları güncellemeyi kapatır
    print("Model başarıyla yüklendi! Kameraya bağlanılıyor...")
    
    # MediaPipe Yüz Algılayıcı (Tasks API) Kurulumu
    from mediapipe.tasks import python
    from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
    
    base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
    options = FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
    
    # Web Kamerasını Aç
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("HATA: Kamera algılanamadı veya açılamadı.")
        return
        
    # Modelin eğitiminde kullanılan veriyi modele sokma kuralları
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    print("--- Canlı Yüz Tanıma Aktif ---")
    print("Çıkmak için ekrandaki pencereye tıklayıp klavyeden 'q' tuşuna basın.")
    
    with FaceDetector.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # MediaPipe RGB formatında hesaplama yapar, OpenCV'den gelen BGR görüntüyü çevirelim
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            results = detector.detect(mp_image)
            
            # Modelimiz sadece Siyah/Beyaz çalışıyor, kesilecek alanlar için GRİ referans alalım
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Eğer sahnede yüz(ler) bulunduysa
            if results.detections:
                for detection in results.detections:
                    
                    # Yeni MediaPipe Tasks API, direkt olarak gerçek piksel koordinatları (x, y, w, h) verir
                    bbox = detection.bounding_box
                    x = int(bbox.origin_x)
                    y = int(bbox.origin_y)
                    w = int(bbox.width)
                    h = int(bbox.height)
                    
                    # Eğer kafa kameranın dışına yarı taştıysa negatif piksel okumasını engelle (Çökme koruması)
                    x, y = max(0, x), max(0, y)
                    
                    # 1. Aşama: Bulunan yüzü kare olarak kırp
                    roi_gray = gray_frame[y:y+h, x:x+w]
                    
                    # Yüz kutusu aşırı ufaldıysa veya hata varsa işlem yapma
                    if roi_gray.shape[0] == 0 or roi_gray.shape[1] == 0:
                        continue
                    
                    # 2. Aşama: Tensora çevirip yapay zekaya (Custom CNN) yollama
                    # Görsel şekli (1, 1, 48, 48) => (Batch_Size, Channel_Grayscale, Height, Width)
                    tensor_img = transform(roi_gray).unsqueeze(0).to(device) 
                    
                    # 3. Aşama: AI Analizi
                    with torch.no_grad(): # Gradient (geri yayılım) hesaplama (RAM/CPU tasarrufu)
                        output = model(tensor_img)
                        _, predicted = torch.max(output.data, 1)
                        emotion_idx = predicted.item()
                        emotion_label = idx_to_emotion.get(emotion_idx, "Bilinmeyen")
                        
                    # 4. Aşama: Sonucu kamerada kutu ve yazı ile gösterme
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, emotion_label.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            
            # Görüntüyü ekrana yansıt
            cv2.imshow('VibeRythm Gercek Zamanli Duygu Analizi (MediaPipe)', frame)
            
            # Çıkış işlemi
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
