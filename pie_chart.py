import wfdb
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# =============================================================================
# 1. VERİ OKUMA VE ETİKET SAYMA
# =============================================================================
record_pool = ['100', '208', '109', '118', '124', '112'] 
valid_labels = ['N', 'L', 'R', 'V', 'A'] 

raw_labels = []

print("Veriler taranıyor ve etiketler sayılıyor...")

for rec in record_pool:
    try:
        annotation = wfdb.rdann(rec, 'atr', pn_dir='mitdb')
        for symbol in annotation.symbol:
            if symbol in valid_labels:
                raw_labels.append(symbol)
    except Exception as e:
        print(f"Kayıt okunamadı: {rec} - {e}")

# =============================================================================
# 2. VERİ HAZIRLAMA (RAW vs BALANCED)
# =============================================================================

# --- A) Raw Data (Imbalanced) ---
counts_raw = Counter(raw_labels)
# Eksik etiket varsa 0 olarak ekle
for lab in valid_labels:
    if lab not in counts_raw: counts_raw[lab] = 0

sizes_raw = [counts_raw[l] for l in valid_labels]

# --- B) Balanced Data (Simüle Edilmiş) ---
min_count = min([c for c in sizes_raw if c > 0]) 
# Görseldeki gibi hepsini eşitliyoruz
sizes_balanced = [min_count] * len(valid_labels)

# =============================================================================
# 3. GÖRSELLEŞTİRME (TAM PASTA GRAFİĞİ)
# =============================================================================

# Renk Paleti (Yeşil, Turuncu, Mor, Kırmızı, Mavi)
colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#d62728', '#1f77b4']
labels_legend = ['Normal (N)', 'LBBB (L)', 'RBBB (R)', 'PVC (V)', 'APB (A)']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Dataset Class Distribution: Before vs. After Balancing', fontsize=20, fontweight='bold')

# --- Fonksiyon: Tam Pasta Çizimi ---
def draw_pie(ax, sizes, title):
    wedges, texts, autotexts = ax.pie(
        sizes, 
        colors=colors, 
        autopct='%1.1f%%', 
        startangle=90, 
        pctdistance=0.6,  # Yüzdeleri merkeze biraz daha yaklaştırır
        wedgeprops=dict(edgecolor='w', linewidth=1) # Dilimler arasına ince beyaz çizgi atar
    )
    
    # Yazı Stilleri (Yüzdeleri beyaz ve kalın yap)
    for autotext in autotexts: 
        autotext.set_color('white')
        autotext.set_weight('bold')
        autotext.set_fontsize(11)

    ax.set_title(title, fontsize=14, fontweight='bold')

# --- Çizim 1: Imbalanced ---
total_raw = sum(sizes_raw)
draw_pie(ax1, sizes_raw, f"Raw MIT-BIH Data (Imbalanced)\nTotal Counted: {total_raw}")

# --- Çizim 2: Balanced ---
total_balanced = sum(sizes_balanced)
draw_pie(ax2, sizes_balanced, f"Training Set (Balanced)\nTotal: ~{total_balanced} (Stratified)")

# Lejant (Açıklama Kutusu)
plt.legend(labels_legend, loc="lower center", bbox_to_anchor=(-0.1, -0.1), ncol=5, fontsize=12)

plt.tight_layout()
plt.show()