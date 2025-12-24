import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import skew, kurtosis, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

# --- CONFIGURATION ---
# MIT-BIH data all good records
all_records = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115', '116', '117', '118', '119', '121', '122', '123', '124', 
    '200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234'
]

records_to_use = all_records 
window_size = 90  
fs = 360          # sampling frequency

all_features = []
all_labels = []

print(f"Processing a total of {len(records_to_use)} patient records... Please wait.")

# --- STEP 1 & 2: LOAD DATA, FILTERING & FEATURE EXTRACTION ---

def apply_filter(data, fs):
    # Bandpass filter (0.5 - 45 Hz) to remove noise
    low = 0.5 / (0.5 * fs)
    high = 45 / (0.5 * fs)
    b, a = signal.butter(2, [low, high], btype='band')
    return signal.lfilter(b, a, data)

for rec_name in records_to_use:
    try:
        # Download the data from PhysioNet
        record = wfdb.rdrecord(rec_name, pn_dir='mitdb')
        annotation = wfdb.rdann(rec_name, 'atr', pn_dir='mitdb')
        
        # Apply Filter
        clean_sig = apply_filter(record.p_signal[:, 0], fs)
        
        for i in range(len(annotation.sample)):
            peak = annotation.sample[i]
            label_symbol = annotation.symbol[i]
            
            # Filter symbols: Keep Normal (N) and common Arrhythmias (L, R, V, A)
            if label_symbol not in ['N', 'L', 'R', 'V', 'A']:
                continue
            
            # Labeling: N -> Normal, Others -> Arrhythmia
            final_label = 'Normal' if label_symbol == 'N' else 'Arrhythmia'
            
            # Boundary checks
            if peak < window_size or peak > len(clean_sig) - window_size:
                continue
                
            # Segmentation
            segment = clean_sig[peak - window_size : peak + window_size]
            
            # --- FEATURE EXTRACTION (The Mathematical Part) ---
            f_mean = np.mean(segment)
            f_std = np.std(segment)
            f_max = np.max(segment)
            f_min = np.min(segment)
            f_skew = skew(segment)
            f_kurt = kurtosis(segment)
            
            all_features.append([f_mean, f_std, f_max, f_min, f_skew, f_kurt])
            all_labels.append(final_label)
            
    except Exception as e:
        print(f"Error: Record {rec_name} could not be read. ({e})")
        continue

# Convert to DataFrame
df = pd.DataFrame(all_features, columns=['Mean', 'Std_Dev', 'Max', 'Min', 'Skewness', 'Kurtosis'])
df['Label'] = all_labels

print(f"Process completed. Total segments processed: {len(df)}")
print("-" * 50)

# --- NEW: PRINT MATHEMATICAL SUMMARY (SAMPLE OUTPUTS) ---
# Bu kısım sunumda "Bakın mean ve std hesapladık" dediğin yerin kanıtıdır.
print("DATASET STATISTICAL SUMMARY (Grouped by Class):")
summary_table = df.groupby('Label')[['Mean', 'Std_Dev', 'Skewness', 'Kurtosis']].agg(['mean', 'std'])
print(summary_table)
print("-" * 50)
print("FIRST 5 ROWS OF EXTRACTED FEATURES:")
print(df.head())
print("-" * 50)


# --- STEP 3: GRAPHS ---

# Figure 1: Data Info - Pie Chart
plt.figure(figsize=(7, 7))
df['Label'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=90, explode=(0.1, 0))
plt.title("Dataset Distribution: Imbalanced Data Problem")
plt.ylabel("")
plt.show()
print(f"Figure 1: Data Imbalance (Pie Chart)")

# Figure 2: Histogram - Std_Dev Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Std_Dev', hue='Label', kde=True, element="step", palette=['blue', 'red'])
plt.title("Distribution of Standard Deviation (Normal vs Arrhythmia)")
plt.xlabel("Standard Deviation")
plt.show()
print(f"Figure 2: Histogram of Standard Deviation")

# Figure 3: Histogram - Skewness Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Skewness', hue='Label', kde=True, element="step", palette=['blue', 'red'])
plt.title("Distribution of Skewness (Feature Analysis)")
plt.xlabel("Skewness Value")
plt.xlim(-2, 2) # Odaklanmak için aralığı daraltabiliriz
plt.show()
print(f"Figure 3: Histogram of Skewness")

# Figure 4: Box Plot - Kurtosis & Outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x='Label', y='Kurtosis', data=df, palette=['blue', 'red'])
plt.title("Box Plot Analysis: Kurtosis & Outliers")
plt.show()
print(f"Figure 4: Box Plot of Kurtosis")

# --- STATISTICAL TEST (T-Test) ---
normal_std = df[df['Label']=='Normal']['Std_Dev']
arrhythmia_std = df[df['Label']=='Arrhythmia']['Std_Dev']
t_stat, p_val = ttest_ind(normal_std, arrhythmia_std)

print("\n")
print(f"T-Test Result (Hypothesis Testing for Std_Dev):")
print(f"P-value: {p_val:.10f}")
if p_val < 0.05:
    print("Result: There is a statistically significant difference (H0 rejected).")
else:
    print("Result: No significant difference.")
print("\n")


# --- STEP 4: MACHINE LEARNING (Logistic Regression) ---
X = df.drop('Label', axis=1)
y = df['Label'].apply(lambda x: 0 if x == 'Normal' else 1) # 0: Normal, 1: Arrhythmia

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Model: Logistic Regression with Class Weight Balanced
# 'class_weight="balanced"' parametresi düşük recall sorununu çözer!
model = LogisticRegression(max_iter=1000, class_weight='balanced') 
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Figure 5: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Arrhythmia'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Final Result)")
plt.show()
print("Figure 5: Confusion Matrix")

# Figure 6: ROC Curve (Optional but good for score)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
print(f"Figure 6: ROC Curve (AUC = {roc_auc:.2f})")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Arrhythmia']))