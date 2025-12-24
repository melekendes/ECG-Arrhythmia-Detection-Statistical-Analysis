import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =============================================================================
# 1. AYARLAR (Gerçek MIT-BIH Hasta Numaraları)
# =============================================================================
# MIT-BIH veritabanındaki hasta ID'leri
records = ['100', '109', '111', '118', '124', '214', '200']

# Görmek istediğimiz sınıflar
target_classes = {
    'N': 'Normal Beat',
    'L': 'LBBB (Left Bundle Block)',
    'R': 'RBBB (Right Bundle Block)',
    'V': 'PVC (Ventricular Contraction)',
    'A': 'APB (Atrial Premature)'
}

# Sunum için net renkler
colors = {'N': 'green', 'L': 'orange', 'R': 'purple', 'V': 'red', 'A': 'blue'}

# =============================================================================
# 2. VERİ ÇEKME VE İŞLEME
# =============================================================================
found_signals = {}
print("Gerçek veriler PhysioNet sunucusundan çekiliyor... (Biraz bekletebilir)")

for rec in records:
    if len(found_signals) == len(target_classes): break # Hepsini bulduysak dur
    
    try:
        # İnternetten veriyi indir
        record = wfdb.rdrecord(rec, pn_dir='mitdb')
        annotation = wfdb.rdann(rec, 'atr', pn_dir='mitdb')
        
        # Sinyali temizle (Filtrele)
        raw = record.p_signal[:, 0]
        b, a = signal.butter(2, [0.5 / 180, 45 / 180], btype='band')
        clean_sig = signal.lfilter(b, a, raw)
        
        # Atımları kontrol et
        for i, sym in enumerate(annotation.symbol):
            if sym in target_classes and sym not in found_signals:
                peak = annotation.sample[i]
                # Kenarlarda değilse al
                if peak > 100 and peak < len(clean_sig)-100:
                    found_signals[sym] = clean_sig[peak-90 : peak+90] # 180 örnek al
                    print(f"-> Bulundu: {target_classes[sym]}")
    except Exception as e:
        continue # Bu hasta inmezse diğerine geç

# =============================================================================
# 3. ÇİZİM (DİREKT EKRANA)
# =============================================================================
if not found_signals:
    print("\n!!! HATA: İnternet bağlantısı veya sunucu sorunu nedeniyle veri çekilemedi.")
else:
    print("\nGrafik çiziliyor...")
    plt.figure(figsize=(10, 10))
    plt.suptitle('Real MIT-BIH Arrhythmia Samples', fontsize=16, fontweight='bold')

    order = ['N', 'V', 'L', 'R', 'A'] # Sıralama

    for i, key in enumerate(order):
        plt.subplot(5, 1, i+1)
        if key in found_signals:
            plt.plot(found_signals[key], color=colors[key], linewidth=2.5)
            plt.title(target_classes[key], loc='left', fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.ylabel("mV")
            plt.xlim(0, 180)
            # Sadece en alttakine x ekseni koy
            if i < 4: plt.xticks([]) 
        else:
            plt.text(90, 0, "Veri Yok", ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()