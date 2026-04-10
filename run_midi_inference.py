"""
1. Başlangıç Notaları (Pitch Numbers - 60, 36, 38 vb.)
instruments_setup içindeki numaralar (örneğin 60), enstrümanın çalmaya başlayacağı ilk notanın frekansıdır (oktavıdır). Yapay zeka ilk notayı buradan alır, gerisini kendisi hayal edip uydurmaya devam eder.

60 (Middle C / Orta Do): Çoğu enstrüman (Piyano, Elektro Gitar, Flüt) için ideal insan kulağı orta seviyesidir.
48 (C3 / 1 Oktav Alt): Çello gibi daha tok ve derin sesler için.
36 (C2 / 2 Oktav Alt): Bas Gitar veya kontrbas için. (Bas gitara 60 verirsek incecik keman gibi ses çıkarır, bu yüzden 36'dan başlattım).
72 (C5 / 1 Oktav Üst): Keman, Piccolo veya ince Synthesizer lead sesleri için.
Bateri Notu (Sadece Drums kanalında geçerlidir): Bateride notalar do-re-mi değildir; davul parçalarıdır. (35 veya 36: Kick Davulu, 38: Trampet/Snare, 42: Hi-Hat).

2. Enstrüman İsimleri (Tırnak İçine Yazdığımız İsimler)
Tırnak içine (örneğin 'Distortion Guitar') tam olarak doğru ismi İngilizce ve standarda uygun yazman lazım.

Gitarlar ve Baslar
'Acoustic Guitar (nylon)' veya 'Acoustic Guitar (steel)'
'Electric Guitar (clean)', 'Electric Guitar (jazz)' veya 'Electric Guitar (muted)'
'Overdriven Guitar' veya 'Distortion Guitar'
'Acoustic Bass'
'Electric Bass (finger)' veya 'Electric Bass (pick)'
'Fretless Bass' veya 'Slap Bass 1'

Piyanolar ve Tuşlular
'Acoustic Grand Piano'
'Electric Piano 1' veya 'Electric Piano 2'
'Harpsichord' (Klavsen)
'Clavinet'
'Church Organ'
'Accordion'

Yaylılar (Strings)
'Violin' (Keman)
'Viola' (Viyola)
'Cello'
'Contrabass'
'String Ensemble 1' (Keman Orkestrası)

Üflemeliler (Brass / Woodwinds)
'Trumpet'
'Trombone'
'French Horn'
'Flute'
'Alto Sax'

Synthesizer / Elektronik
'Lead 1 (square)', 'Lead 2 (sawtooth)'
'Pad 1 (new age)', 'Pad 2 (warm)', 'Pad 3 (polysynth)'
'FX 1 (rain)', 'FX 2 (soundtrack)'
"""

import os
import json
import torch
import pretty_midi
import argparse
from train_midi_model import MidiEmotionTFT
from prepare_midi_model_data import BASE_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    """Config dosyasından TFT parametrelerini okuyup modeli yükler."""
    # Varsayılan TFT parametreleri
    d_model    = 64
    hidden_dim = 256
    num_layers = 2
    num_heads  = 4
    dropout    = 0.1

    model_dir  = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)

    config_path = None
    if model_name == "EN_IYI_MODEL.pth":
        config_path = os.path.join(model_dir, "EN_IYI_MODEL_config.json")
    elif os.path.exists(os.path.join(model_dir, "config.json")):
        config_path = os.path.join(model_dir, "config.json")

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        d_model    = cfg.get("d_model",    d_model)
        hidden_dim = cfg.get("hidden_dim", hidden_dim)
        num_layers = cfg.get("num_layers", num_layers)
        num_heads  = cfg.get("num_heads",  num_heads)
        dropout    = cfg.get("dropout",    dropout)
        print(f"[*] Config yuklendi: d={d_model}, H={hidden_dim}, L={num_layers}, heads={num_heads}, drop={dropout}")
    else:
        print("[!] Config bulunamadi, varsayilan TFT parametreleri kullaniliyor.")

    model = MidiEmotionTFT(
        emotion_classes=4,
        pitch_classes=128,
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def generate_midi(model_path, emotion_idx, num_notes=50, out_file="generated.mid"):
    # 1. Modeli Yükle
    model = load_model(model_path)

    emotion_tensor = torch.tensor([emotion_idx], dtype=torch.long).to(device)

    duygular = {
        0: "Q1 - Neseli/Mutlu",
        1: "Q2 - Gergin/Korkulu",
        2: "Q3 - Uzgun/Melankolik",
        3: "Q4 - Huzurlu/Rahat",
    }
    print(f"\n[*] TFT Orkestra Uretimi Basladi: {duygular.get(emotion_idx, 'Bilinmeyen Duygu')}")

    instruments_setup = {
        'Acoustic Guitar (nylon)': 36,
        'Electric Bass (finger)': 36,
    }

    midi = pretty_midi.PrettyMIDI()

    for instr_name, start_pitch in instruments_setup.items():
        print(f"  -> Uretiliyor: {instr_name}")
        current_notes = [[start_pitch, 80, 0.5]]

        with torch.no_grad():
            for i in range(num_notes):
                # (1, T, 3) tensor
                notes_tensor = torch.tensor([current_notes], dtype=torch.float32).to(device)

                pitch_logits, vel_preds, dur_preds = model(notes_tensor, emotion_tensor)

                # Sadece en son zaman adımının tahmini bize lazım
                next_pitch_logits = pitch_logits[0, -1, :]
                next_vel  = vel_preds[0, -1].item()
                next_dur  = dur_preds[0, -1].item()

                # --- YARATICILIK: Multinomial örnekleme ---
                probs      = torch.softmax(next_pitch_logits, dim=-1)
                next_pitch = torch.multinomial(probs, 1).item()

                # 3 ardışık aynı nota tekrarını engelle
                if (
                    len(current_notes) >= 2
                    and current_notes[-1][0] == next_pitch
                    and current_notes[-2][0] == next_pitch
                ):
                    top_vals, top_classes = torch.topk(probs, 3)
                    for p_class in top_classes:
                        if p_class.item() != next_pitch:
                            next_pitch = p_class.item()
                            break

                # Sınır klempleme
                next_vel = max(60, min(127, int(next_vel)))
                next_dur = max(0.2, min(3.0, next_dur))

                current_notes.append([next_pitch, next_vel, next_dur])

        # 3. Nota listesini MIDI'ye çevir
        is_drum = (instr_name == 'Drums')
        program = 0 if is_drum else pretty_midi.instrument_name_to_program(instr_name)
        instrument = pretty_midi.Instrument(program=program, is_drum=is_drum)

        current_time = 0.0
        for pitch, vel, dur in current_notes:
            note = pretty_midi.Note(
                velocity=int(vel),
                pitch=int(pitch),
                start=current_time,
                end=current_time + dur,
            )
            instrument.notes.append(note)
            current_time += dur

        midi.instruments.append(instrument)

    midi.write(out_file)
    print(f"\n[*] MIDI dosyasi kaydedildi -> {out_file}")
    print(
        f"[+] TAMAMLANDI: {len(instruments_setup)} enstrumanli TFT orkestrasina ait "
        f"{num_notes} notalik muzik '{out_file}' olarak kaydedildi!\n"
    )

    # 4. Pygame ile doğrudan çal
    try:
        import pygame
        print(">> OLUŞTURULAN TFT MÜZİĞİ ÇALINIYOR... <<")
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(out_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        print("Çalma tamamlandı!")
    except ImportError:
        print("[!] pygame bulunamadı — üretilen MIDI dosyasını elle açabilirsiniz.")
    except Exception as e:
        print(f"[!] Otomatik çalma hatası: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFT Tabanli MIDI Ureticisi")
    parser.add_argument("--model",   type=str, default="EN_IYI_MODEL.pth",
                        help="Uretici model dosyasinin yolu")
    parser.add_argument("--emotion", type=int, default=0,
                        help="0:Q1(Mutlu)  1:Q2(Gergin)  2:Q3(Uzgun)  3:Q4(Huzurlu)")
    parser.add_argument("--length",  type=int, default=150,
                        help="Parca icerisinde basilan toplam nota sayisi")
    parser.add_argument("--out",     type=str, default="VibeRythm_Output.mid",
                        help="Kaydedilecek render dosyasinin adi")

    args = parser.parse_args()

    import glob
    model_full_path = os.path.join(BASE_DIR, args.model)

    if not os.path.exists(model_full_path):
        print(f"[*] Final model bulunamadi ({args.model}). Alternatif aranıyor...")
        autotune_dir = os.path.join(BASE_DIR, "AutoTuneModels")
        if os.path.exists(autotune_dir):
            pth_files = glob.glob(os.path.join(autotune_dir, "Model_*", "*.pth"))
            if pth_files:
                model_full_path = max(pth_files, key=os.path.getctime)
                print(f"[*] AutoTune modeli bulundu -> {os.path.basename(model_full_path)}")
            else:
                old_model = os.path.join(BASE_DIR, "midi_emotion_lstm.pth")
                if os.path.exists(old_model):
                    model_full_path = old_model
                    print("[*] Eski yedek bulundu -> midi_emotion_lstm.pth")

    if not os.path.exists(model_full_path):
        print("\n[!] HATA: Hicbir model dosyasi bulunamadi!")
        print("Lutfen once train_midi_model.py ile egitimi tamamlayin.\n")
    else:
        generate_midi(
            model_full_path,
            args.emotion,
            args.length,
            os.path.join(BASE_DIR, args.out),
        )