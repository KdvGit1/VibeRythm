import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import mediapipe as mp

# Dosya yolları
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "lotr_characters", "input")
AUG_DIR = os.path.join(BASE_DIR, "lotr_characters", "augmented")

def check_input_dir():
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_exts)]
    return files

def get_face_cropper():
    """MediaPipe Yüz Algılayıcı başlatır."""
    from mediapipe.tasks import python
    from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
    
    model_path = os.path.join(BASE_DIR, 'blaze_face_short_range.tflite')
    if not os.path.exists(model_path):
        print(f"HATA: '{model_path}' bulunamadı.")
        sys.exit(1)
        
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.4)
    return FaceDetector.create_from_options(options)

def augment_image(face_pil, num_samples=100):
    """Resmi döndürme, parlaklık, kontrast, kesme vb. yöntemlerle çoğaltır."""
    # HOG (Histogram of Oriented Gradients) mantığına benzer şekilde, 
    # modelin kenar, yön ve ışık değişimlerine dayanıklı olması için 
    # aşağıdaki veri çeşitlendirme (data augmentation) adımlarını uyguluyoruz.
    augmenter = T.Compose([
        T.RandomHorizontalFlip(p=0.5), # Yatay çevirme
        T.RandomRotation(degrees=(-15, 15), interpolation=T.InterpolationMode.BILINEAR), # Döndürme
        T.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.05), ratio=(0.9, 1.1)), # Ölçekleme ve kesme
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05), # Işık ve kontrast değişimleri
        T.RandomAffine(degrees=0, translate=(0.04, 0.04)), # Kaydırma
    ])
    
    augmented_images = []
    for _ in range(num_samples):
        aug_img = augmenter(face_pil)
        augmented_images.append(aug_img)
        
    return augmented_images

def main():
    print("--- Karakter Yüzü Çoğaltma (Data Augmentation) Başlatılıyor ---")
    
    input_files = check_input_dir()
    if not input_files:
        print("\n[UYARI] 'lotr_characters/input/' klasöründe fotoğraf bulunamadı!")
        print("Lütfen videodaki karakterlerin yüzlerinin 1'er adet fotoğrafını bu klasöre yükleyin.")
        print("Örnek dosya isimleri: 'frodo.jpg', 'gandalf.png', 'aragorn.jpeg'")
        return
        
    print(f"Bulunan karakter dosyaları: {input_files}")
    
    # MediaPipe Yüz Algılayıcıyı yükle
    detector = get_face_cropper()
    
    for filename in input_files:
        filepath = os.path.join(INPUT_DIR, filename)
        # Karakter adı dosya isminden alınır (örn: frodo.jpg -> frodo)
        char_name = os.path.splitext(filename)[0].lower().strip()
        
        # Karakter için çıktı klasörünü oluştur
        char_aug_dir = os.path.join(AUG_DIR, char_name)
        os.makedirs(char_aug_dir, exist_ok=True)
        
        # Resmi yükle
        bgr_img = cv2.imread(filepath)
        if bgr_img is None:
            print(f"[HATA] '{filename}' okunamadı, geçiliyor...")
            continue
            
        h, w, c = bgr_img.shape
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        # Yüz tespiti yap
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        detection_result = detector.detect(mp_image)
        
        face_cropped = None
        if detection_result.detections:
            # En büyük/ilk yüzü seç
            detection = detection_result.detections[0]
            bbox = detection.bounding_box
            
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            width = int(bbox.width)
            height = int(bbox.height)
            
            # Negatif veya dışarı taşan koordinatları sınırla
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + width)
            y2 = min(h, y + height)
            
            face_cropped = rgb_img[y1:y2, x1:x2]
            print(f"'{char_name}' için yüz başarıyla tespit edildi ve kırpıldı.")
        else:
            print(f"[UYARI] '{char_name}' fotoğrafında yüz tespit edilemedi. Tüm görsel kullanılacak.")
            face_cropped = rgb_img
            
        # PIL Görseline dönüştür ve 224x224 boyutlandır
        face_pil = Image.fromarray(face_cropped)
        face_pil_resized = face_pil.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Orijinal kırpılmış yüzü kaydet
        face_pil_resized.save(os.path.join(char_aug_dir, f"{char_name}_original.jpg"), "JPEG")
        
        # 100 Adet çoğaltılmış resim üret
        print(f"'{char_name}' için 100 adet çoğaltılmış fotoğraf üretiliyor...")
        aug_images = augment_image(face_pil_resized, num_samples=100)
        
        for i, aug_img in enumerate(aug_images):
            aug_img.save(os.path.join(char_aug_dir, f"{char_name}_aug_{i}.jpg"), "JPEG")
            
        print(f"--> '{char_name}' tamamlandı. Çoğaltılan fotoğraflar kaydedildi: 'lotr_characters/augmented/{char_name}/'")
        
    print("\nTüm karakterlerin veri çoğaltma işlemi başarıyla tamamlandı!")

if __name__ == "__main__":
    main()
