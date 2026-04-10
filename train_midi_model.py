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

# ============================================================
# TFT Yardımcı Bloklar
# ============================================================

class GatedResidualNetwork(nn.Module):
    """
    TFT'nin temel yapı taşı.
    Girdiyi doğrusal dönüşümden geçirir, ELU aktivasyonu uygular,
    GLU kapısıyla filtreler ve artık (residual) bağlantıyla toplar.
    Opsiyonel bir bağlam (context) vektörü statik bilgiyi enjekte eder.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Bağlam (duygu embedding) için isteğe bağlı projeksiyon
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim * 2)   # GLU için 2x çıktı
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Giriş boyutu çıkışla eşleşmiyorsa artık bağlantı için projeksiyon
        self.skip_proj = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else None

    def forward(self, x, context=None):
        residual = self.skip_proj(x) if self.skip_proj else x

        h = self.fc1(x)
        if context is not None and self.context_proj is not None:
            h = h + self.context_proj(context)
        h = torch.nn.functional.elu(h)
        h = self.dropout(h)

        # Gated Linear Unit (GLU)
        h = self.fc2(h)
        h1, h2 = h.chunk(2, dim=-1)
        h = h1 * torch.sigmoid(h2)

        return self.layer_norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Her zaman adımındaki her özelliğe (pitch, velocity, duration) önem ağırlığı atar.
    Softmax tabanlı ağırlıklarla özellikleri ağırlıklı toplar.
    """
    def __init__(self, num_inputs, input_dim, hidden_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.num_inputs = num_inputs

        # Her özellik için ayrı GRN
        self.single_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_inputs)
        ])

        # Ağırlık seçim GRN'si (bağlamla birlikte)
        self.selection_grn = GatedResidualNetwork(
            num_inputs * input_dim, hidden_dim, num_inputs, dropout, context_dim=context_dim
        )

    def forward(self, features, context=None):
        # features: (B, T, num_inputs, input_dim)  ya da  (B, num_inputs, input_dim) statik için
        is_temporal = features.dim() == 4
        if is_temporal:
            B, T, N, D = features.shape
            flat = features.reshape(B * T, N * D)
            ctx = context.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1) if context is not None else None
        else:
            B, N, D = features.shape
            flat = features.reshape(B, N * D)
            ctx = context

        weights = torch.softmax(self.selection_grn(flat, ctx), dim=-1)  # (..., N)

        # Her özellik için GRN uygula
        processed = []
        for i, grn in enumerate(self.single_grns):
            fi = features[..., i, :].reshape(-1, features.shape[-1]) if is_temporal else features[:, i, :]
            processed.append(grn(fi))  # (..., hidden_dim)

        processed = torch.stack(processed, dim=-2)  # (..., N, hidden_dim)
        weights = weights.unsqueeze(-1)               # (..., N, 1)
        out = (processed * weights).sum(dim=-2)       # (..., hidden_dim)

        if is_temporal:
            out = out.reshape(B, T, -1)

        return out


class TemporalSelfAttention(nn.Module):
    """
    TFT'nin yorumlanabilir çok-başlı dikkat (interpretable multi-head attention) katmanı.
    Tüm başlar paylaşılan V matrisini kullanır; K ve Q başa özeldir.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_head)  # Paylaşılan V

        self.out_proj = nn.Linear(self.d_head, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        H, Dh = self.num_heads, self.d_head

        Q = self.W_q(x).reshape(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)
        K = self.W_k(x).reshape(B, T, H, Dh).transpose(1, 2)
        V = self.W_v(x).unsqueeze(1).expand(B, H, T, Dh)       # Paylaşılan V

        attn = (Q @ K.transpose(-2, -1)) * self.scale           # (B, H, T, T)

        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V)                                         # (B, H, T, Dh)
        out = out.mean(dim=1)                                    # (B, T, Dh) — başlar ortalaması
        return self.out_proj(out), attn.mean(dim=1)              # attn yorumlanabilirlik için


# ============================================================
# Ana TFT Modeli
# ============================================================

class MidiEmotionTFT(nn.Module):
    """
    MIDI duygu üretimi için Temporal Fusion Transformer (TFT).

    Mimari akışı:
      1. Statik kodlayıcı  — duygu embedding'ini GRN ile işler
      2. Girdi embedding    — (pitch, velocity, duration) ayrı linear katmanlarla d_model'e taşınır
      3. Değişken seçimi    — VSN ile her özelliğe ağırlık atanır
      4. Yerel işleme       — LSTM encoder-decoder (TFT standardı)
      5. Statik zenginleştirme — LSTM çıktısı statik bağlamla GRN'den geçer
      6. Dikkat             — Yorumlanabilir çok-başlı temporal self-attention
      7. Konum besleme      — GRN + add&norm
      8. Çıktı kafaları     — pitch (sınıflandırma), velocity & duration (regresyon)
    """

    def __init__(
        self,
        emotion_classes=4,
        pitch_classes=128,
        d_model=64,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # --- Statik Kodlayıcı ---
        self.emotion_emb = nn.Embedding(emotion_classes, d_model)
        self.static_grn = GatedResidualNetwork(d_model, hidden_dim, d_model, dropout)

        # Statik bağlamlar (LSTM başlangıç durumu + zenginleştirme)
        self.static_context_h = GatedResidualNetwork(d_model, hidden_dim, hidden_dim * num_layers, dropout)
        self.static_context_c = GatedResidualNetwork(d_model, hidden_dim, hidden_dim * num_layers, dropout)
        self.static_context_enrich = GatedResidualNetwork(d_model, hidden_dim, d_model, dropout)

        # --- Girdi Embedding ---
        self.pitch_emb  = nn.Embedding(pitch_classes, d_model)
        self.vel_proj   = nn.Linear(1, d_model)
        self.dur_proj   = nn.Linear(1, d_model)

        # --- Değişken Seçim Ağı ---
        self.vsn = VariableSelectionNetwork(
            num_inputs=3,
            input_dim=d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            context_dim=d_model,
        )

        # --- Yerel İşleme: LSTM ---
        self.num_layers  = num_layers
        self.hidden_dim  = hidden_dim
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.lstm_proj     = nn.Linear(hidden_dim, d_model)
        self.lstm_gate     = GatedResidualNetwork(d_model, hidden_dim, d_model, dropout)
        self.lstm_norm     = nn.LayerNorm(d_model)

        # --- Statik Zenginleştirme ---
        self.static_enrich = GatedResidualNetwork(d_model, hidden_dim, d_model, dropout, context_dim=d_model)

        # --- Temporal Self-Attention ---
        self.attn          = TemporalSelfAttention(d_model, num_heads, dropout)
        self.attn_gate     = GatedResidualNetwork(d_model, hidden_dim, d_model, dropout)
        self.attn_norm     = nn.LayerNorm(d_model)

        # --- Konum Besleme (Position-wise Feed-forward) ---
        self.ff_grn        = GatedResidualNetwork(d_model, hidden_dim, d_model, dropout)
        self.ff_norm       = nn.LayerNorm(d_model)

        # --- Çıktı Kafaları ---
        self.dropout       = nn.Dropout(dropout)
        self.fc_pitch      = nn.Linear(d_model, pitch_classes)
        self.fc_velocity   = nn.Linear(d_model, 1)
        self.fc_duration   = nn.Linear(d_model, 1)

    # ----------------------------------------------------------
    def forward(self, notes, emotions):
        """
        notes   : (B, T, 3)  — [pitch, velocity, duration]
        emotions: (B,)       — duygu sınıf indeksi
        """
        B, T, _ = notes.shape

        # ---- 1. Statik kodlama ----
        emo_static = self.emotion_emb(emotions)                      # (B, d_model)
        emo_static = self.static_grn(emo_static)                     # (B, d_model)

        # LSTM başlangıç gizli durumları (num_layers, B, hidden_dim)
        h0 = self.static_context_h(emo_static).reshape(self.num_layers, B, self.hidden_dim)
        c0 = self.static_context_c(emo_static).reshape(self.num_layers, B, self.hidden_dim)

        # Statik zenginleştirme bağlamı
        static_enrich_ctx = self.static_context_enrich(emo_static)   # (B, d_model)

        # ---- 2. Girdi embedding ----
        pitches    = notes[:, :, 0].long()
        velocities = (notes[:, :, 1] / 127.0).unsqueeze(-1)
        durations  = notes[:, :, 2].unsqueeze(-1)

        p_emb = self.pitch_emb(pitches)                              # (B, T, d_model)
        v_emb = self.vel_proj(velocities)                            # (B, T, d_model)
        d_emb = self.dur_proj(durations)                             # (B, T, d_model)

        # ---- 3. Değişken seçimi ----
        features = torch.stack([p_emb, v_emb, d_emb], dim=2)        # (B, T, 3, d_model)
        vsn_out  = self.vsn(features, context=emo_static)            # (B, T, hidden_dim)

        # ---- 4. LSTM yerel işleme ----
        lstm_out, _ = self.lstm(vsn_out, (h0, c0))                  # (B, T, hidden_dim)
        lstm_out    = self.lstm_proj(lstm_out)                       # (B, T, d_model)

        # Kapılı artık bağlantı + normalize
        lstm_out = self.lstm_norm(self.lstm_gate(lstm_out) + vsn_out[:, :, :self.d_model])

        # ---- 5. Statik zenginleştirme ----
        ctx_expanded = static_enrich_ctx.unsqueeze(1).expand(B, T, -1)
        enriched = self.static_enrich(lstm_out, context=ctx_expanded)

        # ---- 6. Temporal self-attention ----
        # Nedensellik maskesi: gelecek konumlara bakma
        causal_mask = torch.triu(
            torch.ones(T, T, device=notes.device, dtype=torch.bool), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        attn_out, _ = self.attn(enriched, mask=causal_mask)
        attn_out    = self.attn_norm(self.attn_gate(attn_out) + enriched)

        # ---- 7. Konum besleme ----
        ff_out = self.ff_norm(self.ff_grn(attn_out) + attn_out)
        ff_out = self.dropout(ff_out)

        # ---- 8. Çıktı kafaları ----
        pitch_logits    = self.fc_pitch(ff_out)                      # (B, T, 128)
        velocity_preds  = self.fc_velocity(ff_out).squeeze(-1) * 127.0  # (B, T)
        duration_preds  = self.fc_duration(ff_out).squeeze(-1)          # (B, T)

        return pitch_logits, velocity_preds, duration_preds


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter   = 0

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# ============================================================
# Tek Model Eğitimi
# ============================================================

def train_one_model(config, model_dir, train_loader, val_loader, device):
    print(f"\n[>>>] TFT MODEL EGITILIYOR: {config}")

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    model = MidiEmotionTFT(
        d_model    =config['d_model'],
        hidden_dim =config['hidden_dim'],
        num_layers =config['num_layers'],
        num_heads  =config['num_heads'],
        dropout    =config['dropout'],
    ).to(device)

    criterion_pitch      = nn.CrossEntropyLoss()
    criterion_continuous = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.001))

    # Öğrenme oranı azaltıcı (warmup yok; basit plateau scheduler)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    max_epochs   = 2000
    early_stopper = EarlyStopping(patience=10, min_delta=0.01)
    best_val_loss = float('inf')

    for epoch in range(1, max_epochs + 1):
        # --- Eğitim ---
        model.train()
        total_train_loss = 0.0

        for batch_notes, batch_emotions in train_loader:
            batch_notes    = batch_notes.to(device)
            batch_emotions = batch_emotions.to(device)

            x_notes = batch_notes[:, :-1, :]
            y_notes = batch_notes[:, 1:, :]

            optimizer.zero_grad()
            pitch_logits, vel_preds, dur_preds = model(x_notes, batch_emotions)

            target_pitch = y_notes[:, :, 0].long()
            target_vel   = y_notes[:, :, 1].float()
            target_dur   = y_notes[:, :, 2].float()

            loss_pitch = criterion_pitch(pitch_logits.transpose(1, 2), target_pitch)
            loss_vel   = criterion_continuous(vel_preds, target_vel)
            loss_dur   = criterion_continuous(dur_preds, target_dur)

            total_loss = loss_pitch + (loss_vel * 0.005) + (loss_dur * 0.1)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += total_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Doğrulama ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_notes, batch_emotions in val_loader:
                batch_notes    = batch_notes.to(device)
                batch_emotions = batch_emotions.to(device)

                x_notes = batch_notes[:, :-1, :]
                y_notes = batch_notes[:, 1:, :]

                pitch_logits, vel_preds, dur_preds = model(x_notes, batch_emotions)

                target_pitch = y_notes[:, :, 0].long()
                target_vel   = y_notes[:, :, 1].float()
                target_dur   = y_notes[:, :, 2].float()

                v_loss = (
                    criterion_pitch(pitch_logits.transpose(1, 2), target_pitch)
                    + criterion_continuous(vel_preds, target_vel) * 0.005
                    + criterion_continuous(dur_preds, target_dur) * 0.1
                )
                total_val_loss += v_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        print(f"  Epoch [{epoch:03d}/{max_epochs}] - Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if early_stopper(avg_val_loss):
            print(f"  [!] Early Stopping tetiklendi.")
            break

    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  [*] Kaydedildi: {model_path} | En Iyi Val Loss: {best_val_loss:.4f}")
    return best_val_loss


# ============================================================
# Grid Search / AutoTune
# ============================================================

def grid_search_auto_tune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=======================================================")
    print(f"[*] Cihaz: {str(device).upper()}")
    print("[*] TFT HIPERPARAMETRE GRID SEARCH BASLADI")
    print("=======================================================\n")

    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        print("[*] Egitim verisi bulunamadı. Ilk islem baslatiliyor...")
        prepare_datasets(LABEL_CSV, MIDI_DIR, MIDI_DATA_DIR)

    print("[*] Veriseti yukleniyor...")
    train_dataset = MidiEmotionDataset(pt_data_path=TRAIN_DATA_PATH, seq_length=32)
    val_dataset   = MidiEmotionDataset(pt_data_path=TEST_DATA_PATH,  seq_length=32)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)

    # TFT'ye özel hiperparametre uzayı
    param_grid = {
        'd_model'   : [32, 64, 128],
        'hidden_dim': [128, 256, 512],
        'num_layers': [1, 2, 3],
        'num_heads' : [2, 4],
        'dropout'   : [0.1, 0.3],
        'lr'        : [0.001],
    }

    keys         = list(param_grid.keys())
    values       = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"[*] Toplam {len(combinations)} konfigürasyon test edilecek.")

    AUTOTUNE_DIR = os.path.join(BASE_DIR, "AutoTuneModels")
    os.makedirs(AUTOTUNE_DIR, exist_ok=True)

    model_results = []

    for idx, combo in enumerate(combinations, start=1):
        config    = dict(zip(keys, combo))
        model_dir = os.path.join(AUTOTUNE_DIR, f"Model_{idx}")
        best_val  = train_one_model(config, model_dir, train_loader, val_loader, device)
        model_results.append({"model_id": idx, "folder": model_dir, "config": config, "val_loss": best_val})

    # Skor Tablosu
    print("\n=======================================================")
    print("====== SONUC BILDIRGESI (LEADERBOARD) =================")
    print("=======================================================\n")

    sorted_results = sorted(model_results, key=lambda x: x["val_loss"])

    for i, res in enumerate(sorted_results, start=1):
        c = res["config"]
        print(
            f"{i}. Model_{res['model_id']} | Loss: {res['val_loss']:.4f} | "
            f"d={c['d_model']}, H={c['hidden_dim']}, L={c['num_layers']}, "
            f"heads={c['num_heads']}, drop={c['dropout']}"
        )

    # En iyi modeli kopyala
    best          = sorted_results[0]
    best_src      = os.path.join(best["folder"], "model.pth")
    target_path   = os.path.join(BASE_DIR, "EN_IYI_MODEL.pth")
    shutil.copyfile(best_src, target_path)

    cfg_src       = os.path.join(best["folder"], "config.json")
    cfg_target    = os.path.join(BASE_DIR, "EN_IYI_MODEL_config.json")
    if os.path.exists(cfg_src):
        shutil.copyfile(cfg_src, cfg_target)

    print(f"\n[***] KAZANAN: Model_{best['model_id']} | Val Loss: {best['val_loss']:.4f}")
    print(f"[***] TFT beyni kopyalandi -> {target_path}")
    print("[***] Egitim tamamlandi. Inferans icin run_midi_inference.py kullanabilirsiniz.")


if __name__ == "__main__":
    grid_search_auto_tune()