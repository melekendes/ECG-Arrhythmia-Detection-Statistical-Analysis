import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, ConfusionMatrixDisplay
import random
import time
import warnings
from scipy.stats import ttest_ind

# Silence warnings for clean presentation output
warnings.filterwarnings("ignore")

print("SYSTEM INITIALIZING: Logistic Regression Arrhythmia Detection..\n")

# SETUP & DATASET
record_pool = ['100', '208', '109', '118', '124', '112']
# 100: Normal, Others: Various Arrhythmias

print("Dataset: MIT-BIH Arrhythmia Database")
print(f"Patient Records Used: {record_pool}")
print("Objective: Binary Classification (Normal vs. Abnormal/Risk)")

# FUNCTIONS
def apply_filter(data):
    # Band-Pass Filter (0.5 - 45 Hz) for Noise Reduction
    b, a = signal.butter(2, [0.5 / 180, 45 / 180], btype='band')
    return signal.lfilter(b, a, data)

def extract_features(segment):
    # Statistical Feature Extraction
    return [
        np.mean(segment), np.std(segment), np.max(segment), np.min(segment),
        skew(segment), kurtosis(segment), np.sum(segment**2)
    ]

feature_names = ['Mean', 'Std', 'Max', 'Min', 'Skewness', 'Kurtosis', 'Energy']

# DATA LOADING & PROCESSING
X_list = []      # Features
y_list = []      # Labels (0 or 1) healthy vs. arrhythmia
raw_sig_list = []

print("\nProcessing data..")

for rec in record_pool:
    try:
        record = wfdb.rdrecord(rec, pn_dir='mitdb')
        annotation = wfdb.rdann(rec, 'atr', pn_dir='mitdb')
        raw_full_signal = record.p_signal[:, 0]
        clean_full_signal = apply_filter(raw_full_signal)
        
        count = 0
        for i in range(len(annotation.sample)):
            if count > 2000: break # Balanced loading limit per patient
            
            peak = annotation.sample[i]
            label = annotation.symbol[i]
            
            valid_labels = ['N', 'L', 'R', 'V', 'A', '/']
            if label not in valid_labels: continue
            if peak < 90 or peak > len(clean_full_signal) - 90: continue
            
            
            raw_segment = raw_full_signal[peak-90 : peak+90]
            clean_segment = clean_full_signal[peak-90 : peak+90]
            
        
            features = extract_features(clean_segment)
            
            X_list.append(features)
            # Labeling: N -> 0 (Normal), Others -> 1 (Abnormal)
            y_list.append(0 if label == 'N' else 1)
            # Save RAW segment for visualization
            raw_sig_list.append(raw_segment)
            
            count += 1
    except: continue

df = pd.DataFrame(X_list, columns=feature_names)
y = np.array(y_list)
raw_signals_array = np.array(raw_sig_list)

print(f"Total Processed Heartbeats: {len(df)}")
print(f"Normal Class count: {np.sum(y==0)}")
print(f"Abnormal Class count: {np.sum(y==1)}")


#VISUALIZATION
#PREPROCESSING
print("\nMethodology: Signal Preprocessing")
# Grab a specific noisy example for demonstration
demo_idx = np.where(y==1)[0][0] # Grab first abnormal example

plt.figure(figsize=(12, 4))
plt.title("Preprocessing Step: Noise Reduction (Raw vs. Filtered)", fontsize=14)
# Plot RAW signal in background
plt.plot(raw_signals_array[demo_idx], color='gray', alpha=0.5, label='Raw Signal (Noisy & Wandering)')
# Plot FILTERED signal on top
# We need to re-filter this specific raw segment to show the effect directly
demo_filtered = apply_filter(raw_signals_array[demo_idx]) 
plt.plot(demo_filtered, color='blue', linewidth=2, label='Filtered Signal (Clean & Centered)')

plt.xlabel("Time (Samples)")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# #FEATURE ANALYSIS
print("\nFeature Space Analysis")

# Create a temporary dataframe for plotting
df_plot = df.copy()
df_plot['Status'] = ['Normal' if label == 0 else 'Abnormal' for label in y]

# --- PART A: Box Plots (Statistical Distributions) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Statistical Feature Differences (Box Plots)', fontsize=16)
features_to_box = ['Skewness', 'Kurtosis', 'Energy']
colors = ['lightblue', 'lightcoral']

for i, col in enumerate(features_to_box):
    sns.boxplot(x='Status', y=col, data=df_plot, ax=axes[i], palette=colors)
    axes[i].set_title(col, fontweight='bold')
    axes[i].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2D Scatter Plot
plt.figure(figsize=(10, 6))
plt.title("Feature Space Division: Skewness vs. Kurtosis", fontsize=14)
sns.scatterplot(x='Skewness', y='Kurtosis', hue='Status', data=df_plot, 
                palette={'Normal':'blue', 'Abnormal':'red'}, alpha=0.6, style='Status')
plt.grid(True, alpha=0.3)
plt.show()


# STATISTICAL HYPOTHESIS TESTING (T-TEST)
from scipy.stats import ttest_ind

print("\n" + "="*80)
print("STATISTICAL HYPOTHESIS TESTING (SCIENTIFIC PROOF)")
print("="*80)

# DEFINE HYPOTHESES
print("H0 (Null Hypothesis)       : There is NO statistical difference between Normal and Arrhythmia signals.")
print("H1 (Alternative Hypothesis): There IS a significant statistical difference (Discriminative Features).")
print("-" * 80)
print(f"{'Feature':<15} | {'P-Value':<20} | {'Decision (Result)':<40}")
print("-" * 80)

# Features to test
features_to_test = ['Std', 'Skewness', 'Kurtosis', 'Energy']

for feature in features_to_test:
    # Separate groups
    group_normal = df[y == 0][feature]
    group_abnormal = df[y == 1][feature]
    
    # Apply Welch's T-Test (equal_var=False)
    t_stat, p_val = ttest_ind(group_normal, group_abnormal, equal_var=False)
    
    # Decision Logic
    if p_val < 0.05:
        decision = "REJECT H0 -> ACCEPT H1 (Significant)"
    else:
        decision = "FAIL TO REJECT H0 (No Difference)"
        
    print(f"{feature:<15} | {p_val:.5e}          | {decision}")

print("-" * 80)
print("CONCLUSION: Low P-values (<0.05) prove that these features are scientifically valid biomarkers.\n")
print("="*80)

# MODEL TRAINING (LOGISTIC REGRESSION)
# Crucial Step: Split EVERYTHING (Features, Labels, and Raw Signals)
X_train, X_test, y_train, y_test, sig_train_raw, sig_test_raw = train_test_split(
    df, y, raw_signals_array, test_size=0.3, stratify=y, random_state=42
)

# Scaling is mandatory for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining Model (Training Set Size: {len(X_train)})...")
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# RESULTS
print("\nExperimental Results")
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
acc = accuracy_score(y_test, y_pred)

print(f"MODEL ACCURACY (Test Set): %{acc*100:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

# Confusion Matrix Plot
plt.figure(figsize=(5, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Normal', 'Abnormal'], cmap='Blues', ax=plt.gca(), colorbar=False)
plt.title("Confusion Matrix (Test Set)")
plt.show()

# ROC Curve Plot
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve & AUC Score')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# BLIND TEST DEMO WITH ANALYSIS 
print("\nREAL-WORLD BLIND TEST DEMO")

def true_blind_test_demo_advanced():
    # Pick random index from TEST set (Unseen data)
    idx = random.randint(0, len(X_test) - 1)
    
    # Get data for this index
    sample_feat_raw = X_test.iloc[idx] # Original features for bar plot
    sample_feat_scaled = X_test_scaled[idx].reshape(1, -1) # Scaled for model
    actual_label = y_test[idx]
    # Important: Get the FILTERED version for display, as that's what features are based on
    sample_signal_filtered = apply_filter(sig_test_raw[idx]) 
    
    # Prediction
    start = time.time()
    pred = model.predict(sample_feat_scaled)[0]
    dur = (time.time() - start) * 1000
    
    # Text & Colors
    ai_text = "ABNORMAL (RISK)" if pred == 1 else "NORMAL (HEALTHY)"
    real_text = "ABNORMAL" if actual_label == 1 else "NORMAL"
    
    is_correct = (pred == actual_label)
    color = 'green' if is_correct else 'red'
    status = "CORRECT DIAGNOSIS" if is_correct else "MISDIAGNOSIS"
    
    # PLOTTING
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"BLIND TEST SAMPLE #{idx}\nActual: {real_text} | AI Prediction: {ai_text}", 
                 fontsize=16, color=color, fontweight='bold')
    
    # Signal
    ax1.plot(sample_signal_filtered, color='black', linewidth=2)
    ax1.set_title("Processed ECG Signal (Time-Domain)")
    ax1.set_xlabel("Time (Samples)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    
    # Feature Analysis
    # Normalize features for better visual comparison in bar chart (Min-Max scaling just for plot)
    feat_norm = (sample_feat_raw - X_train.min()) / (X_train.max() - X_train.min())
    ax2.bar(feature_names, feat_norm, color=color, alpha=0.6)
    ax2.set_title("Feature Analysis (Normalized Values)")
    ax2.set_ylabel("Relative Feature Importance")
    ax2.grid(True, axis='y', alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
    
    # Status Box
    plt.figtext(0.5, 0.01, f"{status}\nInference Time: {dur:.3f} ms", 
                ha="center", fontsize=12, bbox=dict(fc=color, alpha=0.2))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Make room for suptitle and figtext
    plt.show()

# Run 2 Demos
true_blind_test_demo_advanced()
true_blind_test_demo_advanced()

print("\nPROJECT COMPLETED. All outputs are ready for presentation slides.")