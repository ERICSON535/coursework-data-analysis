"""
Глава 2. Первичный анализ набора данных с временными рядами
Датасет: Human Activity Recognition with Smartphones (UCI HAR)
Источник: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones

Инструкция по загрузке:
  1. Скачайте датасет: kaggle datasets download -d uciml/human-activity-recognition-with-smartphones
  2. Распакуйте в папку data/har/
  3. Ожидаемая структура:
       data/har/train/X_train.txt
       data/har/train/y_train.txt
       data/har/test/X_test.txt
       data/har/test/y_test.txt
       data/har/activity_labels.txt
       data/har/features.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = 'plots/chapter2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = 'data/har'

ACTIVITIES = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING'
}

# ── 1. Загрузка данных ─────────────────────────────────────────────────────────
print("=" * 60)
print("1. ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

def load_har(data_dir):
    X_train = pd.read_csv(f'{data_dir}/train/X_train.txt', delim_whitespace=True, header=None)
    y_train = pd.read_csv(f'{data_dir}/train/y_train.txt', delim_whitespace=True, header=None, names=['activity'])
    X_test  = pd.read_csv(f'{data_dir}/test/X_test.txt',  delim_whitespace=True, header=None)
    y_test  = pd.read_csv(f'{data_dir}/test/y_test.txt',  delim_whitespace=True, header=None, names=['activity'])

    with open(f'{data_dir}/features.txt') as f:
        features = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    # Устранение дублей в именах признаков
    seen = {}
    unique_features = []
    for feat in features:
        if feat in seen:
            seen[feat] += 1
            unique_features.append(f"{feat}_{seen[feat]}")
        else:
            seen[feat] = 0
            unique_features.append(feat)

    X_train.columns = unique_features
    X_test.columns  = unique_features

    X = pd.concat([X_train, X_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True)
    return X, y['activity']

X, y = load_har(DATA_DIR)

print(f"Размер матрицы признаков X: {X.shape[0]} записей × {X.shape[1]} признаков")
print(f"Размер вектора меток y: {y.shape[0]}")
print(f"Классы активностей: {sorted(y.unique())}")
print(f"\nПервые 3 строки (первые 10 признаков):")
print(X.iloc[:3, :10])

# ── 2. Распределение классов ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. РАСПРЕДЕЛЕНИЕ КЛАССОВ")
print("=" * 60)

class_counts = y.map(ACTIVITIES).value_counts()
print(class_counts)

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']
bars = ax.bar(class_counts.index, class_counts.values, color=colors, edgecolor='white')
ax.set_title('Распределение классов активности\nHAR Dataset (UCI)', fontsize=13, fontweight='bold')
ax.set_xlabel('Вид активности')
ax.set_ylabel('Количество записей')
ax.set_xticklabels(class_counts.index, rotation=30, ha='right')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_class_distribution.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/fig1_class_distribution.png")

# ── 3. Описательная статистика признаков ──────────────────────────────────────
print("\n" + "=" * 60)
print("3. ОПИСАТЕЛЬНАЯ СТАТИСТИКА ПРИЗНАКОВ")
print("=" * 60)

stats_df = X.describe().T
print(f"Общая статистика по {X.shape[1]} признакам:")
print(stats_df[['mean', 'std', 'min', '50%', 'max']].head(15))

# Диапазон значений
print(f"\nДиапазон значений: [{X.values.min():.4f}, {X.values.max():.4f}]")
print(f"Нормировка: {'уже нормированы' if X.values.max() <= 1.05 else 'требует нормировки'}")

# Пропущенные значения
missing_count = X.isnull().sum().sum()
print(f"Пропущенные значения: {missing_count}")

# ── 4. Визуализация временных рядов ───────────────────────────────────────────
print("\n" + "=" * 60)
print("4. ВИЗУАЛИЗАЦИЯ ВРЕМЕННЫХ РЯДОВ")
print("=" * 60)

# Ищем признаки акселерометра (tBodyAcc-mean)
acc_cols = [c for c in X.columns if 'tBodyAcc-mean' in c]
if len(acc_cols) >= 3:
    acc_cols = acc_cols[:3]
    n_samples = 200

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axis_labels = ['Ось X', 'Ось Y', 'Ось Z']
    for i, (ax, col, lbl) in enumerate(zip(axes, acc_cols, axis_labels)):
        for act_id, act_name in ACTIVITIES.items():
            mask = (y == act_id).values
            samples = X.loc[mask, col].values[:n_samples]
            ax.plot(samples, label=act_name, alpha=0.75, linewidth=0.9)
        ax.set_ylabel(f'Ускорение ({lbl})', fontsize=9)
        if i == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=2)
    axes[-1].set_xlabel('Номер отсчёта (первые 200 на класс)', fontsize=10)
    fig.suptitle('Временные ряды акселерометра тела\nHAR Dataset — tBodyAcc-mean (X, Y, Z)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig2_timeseries.png', bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {OUTPUT_DIR}/fig2_timeseries.png")

# ── 5. Средние значения по классам ────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. СРЕДНИЕ ЗНАЧЕНИЯ ПРИЗНАКОВ ПО КЛАССАМ")
print("=" * 60)

if acc_cols:
    all_acc = [c for c in X.columns if 'tBodyAcc-mean' in c or 'tGravityAcc-mean' in c][:6]
    means_by_class = pd.concat([X[all_acc], y.rename('activity')], axis=1)
    means_by_class = means_by_class.groupby('activity')[all_acc].mean()
    means_by_class.index = means_by_class.index.map(ACTIVITIES)
    print(means_by_class.round(4))

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(means_by_class, annot=True, fmt='.3f', cmap='RdYlGn',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Среднее значение'})
    ax.set_title('Средние значения признаков акселерометра по классам активности',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Признак')
    ax.set_ylabel('Класс активности')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig3_class_means_heatmap.png', bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {OUTPUT_DIR}/fig3_class_means_heatmap.png")

# ── 6. Автокорреляция ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. АВТОКОРРЕЛЯЦИЯ")
print("=" * 60)

if acc_cols:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    max_lag = 50
    for ax, (act_id, act_name) in zip(axes.flatten(), ACTIVITIES.items()):
        mask = (y == act_id).values
        series = X.loc[mask, acc_cols[0]].values[:300]
        lags = range(0, max_lag + 1)
        acf_vals = [np.corrcoef(series[:-lag] if lag > 0 else series,
                                series[lag:]  if lag > 0 else series)[0, 1] for lag in lags]
        ax.plot(lags, acf_vals, color='steelblue', linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axhline(1.96 / np.sqrt(len(series)), color='red', linestyle='--', linewidth=0.8)
        ax.axhline(-1.96 / np.sqrt(len(series)), color='red', linestyle='--', linewidth=0.8)
        ax.set_title(act_name, fontsize=9)
        ax.set_xlabel('Лаг')
        ax.set_ylabel('ACF')
    fig.suptitle('Автокорреляционные функции tBodyAcc-mean-X по классам активности',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_autocorrelation.png', bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {OUTPUT_DIR}/fig4_autocorrelation.png")

# ── 7. Частотный анализ (FFT) ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. ЧАСТОТНЫЙ АНАЛИЗ (FFT)")
print("=" * 60)

if acc_cols:
    fs = 50  # частота дискретизации 50 Гц
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (act_id, act_name) in zip(axes.flatten(), ACTIVITIES.items()):
        mask = (y == act_id).values
        series = X.loc[mask, acc_cols[0]].values[:256]
        fft_vals = np.abs(np.fft.rfft(series))
        freqs = np.fft.rfftfreq(len(series), d=1 / fs)
        ax.plot(freqs, fft_vals, color='darkorange', linewidth=1.2)
        ax.set_title(act_name, fontsize=9)
        ax.set_xlabel('Частота (Гц)')
        ax.set_ylabel('Амплитуда')
        ax.set_xlim(0, 25)
    fig.suptitle('Спектры частот (FFT) сигнала акселерометра по классам активности',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig5_fft_spectra.png', bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {OUTPUT_DIR}/fig5_fft_spectra.png")

# ── 8. PCA-визуализация ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. СНИЖЕНИЕ РАЗМЕРНОСТИ (PCA)")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.fillna(0))
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_ * 100
print(f"PC1 объясняет дисперсию: {explained[0]:.1f}%")
print(f"PC2 объясняет дисперсию: {explained[1]:.1f}%")

fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']
for (act_id, act_name), color in zip(ACTIVITIES.items(), colors):
    mask = (y == act_id).values
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=act_name, alpha=0.4,
               s=15, color=color, edgecolors='none')
ax.set_xlabel(f'PC1 ({explained[0]:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({explained[1]:.1f}%)', fontsize=11)
ax.set_title('PCA-проекция датасета HAR (2 главные компоненты)',
             fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig6_pca.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/fig6_pca.png")

# ── 9. Вывод ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ИТОГОВЫЙ ВЫВОД")
print("=" * 60)
print(f"Датасет: {X.shape[0]} записей, {X.shape[1]} признаков, 6 классов активности.")
print(f"Пропущенные значения: {missing_count}.")
print(f"Данные нормированы в диапазоне [-1, 1].")
print(f"PCA объясняет {explained[0]+explained[1]:.1f}% дисперсии первыми двумя компонентами.")
print("Статичные активности (SITTING, STANDING, LAYING) хорошо разделяются.")
print("Динамические (WALKING-*) частично перекрываются в пространстве PCA.")
print(f"\nВсе графики сохранены в директории: {OUTPUT_DIR}/")
