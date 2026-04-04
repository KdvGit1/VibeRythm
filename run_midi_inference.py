import os
import torch
import pretty_midi
import argparse
from train_midi_model import MidiEmotionLSTM
from prepare_midi_model_data import BASE_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_midi(model_path, emotion_idx, num_notes=50, out_file="generated.mid"):
    # 1. Modeli Yukle
    model = MidiEmotionLSTM(emotion_classes=4, pitch_classes=128, embed_dim=64, hidden_dim=256, num_layers=2, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # 2. Uretim Icin Ilk Baslangic Notasi 
    # Bosluktan nota uretmek zordur, piyanoda orta Do (Pitch: 60) ile baslatiyoruz. Sirasiyla (Pitch, Velocity, Duration)
    current_notes = [[60, 80, 0.5]]
    
    # Modelin uretecegi duygu hedefini ayarlayalim (Q1:0, Q2:1, Q3:2, Q4:3)
    emotion_tensor = torch.tensor([emotion_idx], dtype=torch.long).to(device)
    
    duygular = {0: "Q1 - Neseli/Mutlu", 1: "Q2 - Gergin/Korkulu", 2: "Q3 - Uzgun/Melankolik", 3: "Q4 - Huzurlu/Rahat"}
    print(f"\n[*] Muzik Uretimi Basladi: {duygular.get(emotion_idx, 'Bilinmeyen Duygu')}")
    
    with torch.no_grad():
        for i in range(num_notes):
            # Formati Tensor yap (1 Batch, x Notalik Gecmis, 3 Ozellik)
            notes_tensor = torch.tensor([current_notes], dtype=torch.float32).to(device)
            
            # Modele gecmisi ve duyguyu gonder
            pitch_logits, vel_preds, dur_preds = model(notes_tensor, emotion_tensor)
            
            # Model diziyi kaydirarak tahmin ettigi icin bize en sondayken atilan son tahmin lazim (siradaki nota)
            next_pitch_logits = pitch_logits[0, -1, :] # 128 genisliginde ihtimaller silsilesi
            next_vel = vel_preds[0, -1].item()
            next_dur = dur_preds[0, -1].item()
            
            # --- YARATICILIK VE TEKRAR KORUMASI ---
            # En yuksek pitch'i aramak yerine Softmax ile "Ihtimaller icerisinden rastgele" oransal cekim yapiyoruz. 
            # (Bu modelin daha insansi ve yaratici calmasini saglar)
            probs = torch.softmax(next_pitch_logits, dim=-1)
            next_pitch = torch.multinomial(probs, 1).item()
            
            # 3 Kere ayni nota tehlikesi var mi? (Egitimde onlemistik ama yine de saglama alalim)
            if len(current_notes) >= 2 and current_notes[-1][0] == next_pitch and current_notes[-2][0] == next_pitch:
                # Eger ayni nota 3. defa rastgeldiyse, siradaki en yuksek 3 ihtimale bak
                top_p, top_class = torch.topk(probs, 3)
                for p_class in top_class:
                    if p_class.item() != next_pitch:
                        next_pitch = p_class.item() # Yeni varyasyonu atayip donguyu kir
                        break
                        
            # --- SINIR YUVALAMALARI ---
            # Egitim basindayken cok dusuk sesler verebiliyor, duymak icin en az 60 ses katsayisi ekliyoruz
            next_vel = max(60, min(127, int(next_vel)))
            next_dur = max(0.2, min(3.0, next_dur)) # Cizirtilari engellemek icin minimum 0.2 saniye uzunluk
            
            
            # Uretilen notayi zincire ekle (Diger adimda gecmis olacak)
            current_notes.append([next_pitch, next_vel, next_dur])
            
    # 3. Listeyi Muzige (.mid) Cevirme Zamani
    print(f"[*] MIDI dosyasi isleniyor -> {out_file}")
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    current_time = 0.0
    for pitch, vel, dur in current_notes:
        note = pretty_midi.Note(
            velocity=int(vel),
            pitch=int(pitch),
            start=current_time,
            end=current_time + dur
        )
        piano.notes.append(note)
        current_time += dur
        
    midi.instruments.append(piano)
    midi.write(out_file)
    print(f"[+] MUTEKIYET VERILDI: {num_notes} notalik muzik basariyla yaratilip '{out_file}' olarak kaydedildi!\n")

    # 4. SESI DIREKT OLARAK YANSITMA (PYGAME ILE)
    try:
        import pygame
        print(">> OLUŞTURULAN YAPAY ZEKA MUZIGI CALINIYOR... <<")
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(out_file)
        pygame.mixer.music.play()
        # Ses bitene kadar bekle
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        print("Calma tamamlandi!")
    except ImportError:
        print("[!] Muzik Python ustunden calinamadi (pygame kütüphanesi eksik).")
        print("Lutfen uretilen dosyaya (VibeRythm_Output.mid) cikan klasorden cift tiklayarak piyanoda dinle.")
    except Exception as e:
        print(f"[!] Otomatik calinirken hata olustu: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trained LSTM MIDI Ureticisi")
    parser.add_argument("--model", type=str, default="EN_IYI_MODEL.pth", help="Uretici model dosyasinin yolu")
    parser.add_argument("--emotion", type=int, default=0, help="Uretilecek Duygu Silsilesi -> 0: Q1(Mutlu), 1: Q2(Gergin), 2: Q3(Uzgun), 3: Q4(Huzurlu)")
    parser.add_argument("--length", type=int, default=150, help="Parca icerisinde basilan toplam nota sayisi")
    parser.add_argument("--out", type=str, default="VibeRythm_Output.mid", help="Kaydedilecek render dosyasinin adi")
    
    args = parser.parse_args()
    
    import glob
    model_full_path = os.path.join(BASE_DIR, args.model)
    
    # Eger ana dosya yoksa, akilli tarama yap
    if not os.path.exists(model_full_path):
        print(f"[*] Onaylanmis final modeli bulunamadi ({args.model}). Alternatif modeller araniyor...")
        if os.path.exists(os.path.join(BASE_DIR, "AutoTuneModels")):
            pth_files = glob.glob(os.path.join(BASE_DIR, "AutoTuneModels", "Model_*", "*.pth"))
            if len(pth_files) > 0:
                model_full_path = max(pth_files, key=os.path.getctime)
                print(f"[*] ALTERNATIF BULUNDU! AutoTune icerisindeki en son uretilen model kullaniliyor -> {os.path.basename(model_full_path)}")
            else:
                # Ana dizindeki eski dosyaya bak
                old_model = os.path.join(BASE_DIR, "midi_emotion_lstm.pth")
                if os.path.exists(old_model):
                    model_full_path = old_model
                    print(f"[*] Eski yedek bulundu -> midi_emotion_lstm.pth")
                    
    # Son denetim
    if not os.path.exists(model_full_path):
        print(f"\n[!] HATA: Hicbir Yapay Zeka Beyni ve Modeli Bulunamadi!")
        print("Lutfen uretimden once egitim asamasini tamamen tamamladiginizdan (train_midi_model.py) emin olun.\n")
    else:
        generate_midi(model_full_path, args.emotion, args.length, os.path.join(BASE_DIR, args.out))
