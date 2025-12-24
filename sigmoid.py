import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit # Sigmoid matematik formülü

print("Veriler project_2.py dosyasından çekiliyor, lütfen bekleyin...")

# =============================================================================
# 1. SENİN PROJENİ İÇE AKTARIYORUZ (IMPORT)
# =============================================================================
try:
    import project_2 as my_project
except ImportError:
    print("HATA: 'project_2.py' dosyası bulunamadı! Bu script ile aynı klasörde olduğundan emin ol.")
    exit()

# =============================================================================
# 2. GEREKLİ VERİLERİ ÇEKİYORUZ
# =============================================================================
# Senin ana kodundaki değişkenleri buraya alıyoruz
model = my_project.model
X_test = my_project.X_test_scaled
y_test = my_project.y_test

print("\nVeriler başarıyla alındı. Sigmoid grafiği çiziliyor...")

# =============================================================================
# 3. SIGMOID (S-EĞRİSİ) GRAFİĞİNİ ÇİZİYORUZ
# =============================================================================
# Adım A: Modelin her hasta için ürettiği "Karar Puanı"nı (z-score) al
decision_scores = model.decision_function(X_test)

# Adım B: Modelin hesapladığı "Olasılık" değerini al (0 ile 1 arası)
probabilities = model.predict_proba(X_test)[:, 1]

# ÇİZİM
plt.figure(figsize=(10, 6))

# 1. Arkadaki Siyah Çizgi (Teorik Sigmoid Eğrisi)
# Bu çizgi modelin matematiksel sınırıdır
range_x = np.linspace(decision_scores.min(), decision_scores.max(), 300)
plt.plot(range_x, expit(range_x), color='black', linestyle='--', linewidth=2, label='Sigmoid Function')

# 2. Senin Gerçek Hastaların (Noktalar)
# Mavi: Normal, Kırmızı: Hasta
colors = np.array(['#3498db' if y == 0 else '#e74c3c' for y in y_test])
plt.scatter(decision_scores, probabilities, c=colors, alpha=0.6, edgecolors='k', zorder=10, s=50)

# 3. Görsel Düzenlemeler
plt.title("Logistic Regression Decision Mechanism (Real Patient Data)", fontsize=14, fontweight='bold')
plt.xlabel("Decision Score", fontsize=12)
plt.ylabel("Probability of Arrhythmia", fontsize=12)

# %50 Eşik Değeri Çizgisi
plt.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='Decision Threshold (0.5)')

# Açıklama Kutusu
plt.text(decision_scores.min(), 0.85, 
         "Blue Dots: Healthy\nRed Dots: Arrhythmia", 
         fontsize=11, bbox=dict(facecolor='white', alpha=0.9))

plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Grafiği Ekrana Bas
plt.show()