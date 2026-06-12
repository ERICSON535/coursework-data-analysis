# -*- coding: utf-8 -*-
"""
Глава 2. Первичный анализ набора данных с временными рядами
Датасет: Power Consumption of Tetouan City
Источник: https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption

Инструкция по загрузке:
  1. Скачайте датасет: kaggle datasets download -d fedesoriano/electric-power-consumption
  2. Распакуйте в папку data/power/
  3. Ожидаемый файл: data/power/powerconsumption.csv

Задача: прогнозирование потребления электроэнергии в трёх зонах города Тетуан
(Марокко) на основе погодных данных (температура, влажность, ветер, радиация).
Тип задачи: регрессия / прогнозирование временных рядов.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

OUTPUT_DIR = 'plots/chapter2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

import matplotlib
matplotlib.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
})

DATA_PATH = 'data/power/powerconsumption.csv'

# ─────────────────────────────────────────────────────────────────
# 1. Загрузка и первичное знакомство с данными
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("2.3.1  Загрузка и общий обзор данных")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime').sort_index()

# Краткие имена столбцов
rename_map = {
    'Temperature': 'temperature',
    'Humidity': 'humidity',
    'Wind Speed': 'wind_speed',
    'general diffuse flows': 'gen_diffuse',
    'diffuse flows': 'diffuse',
    'PowerConsumption_Zone1': 'zone1',
    'PowerConsumption_Zone2': 'zone2',
    'PowerConsumption_Zone3': 'zone3',
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

channels = ['temperature', 'humidity', 'wind_speed', 'gen_diffuse', 'diffuse',
            'zone1', 'zone2', 'zone3']
channels = [c for c in channels if c in df.columns]

print(f"Строки × столбцы: {df.shape}")
print(f"Период: {df.index.min()} — {df.index.max()}")
print(f"Шаг данных: {df.index.to_series().diff().mode()[0]}")
print("\nПервые 3 строки:")
print(df.head(3).to_string())

# ─────────────────────────────────────────────────────────────────
# 2. Визуализация исходных временных рядов
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2.3.2  Визуализация временных рядов")
print("=" * 60)

fig, axes = plt.subplots(len(channels), 1, figsize=(14, 2.5 * len(channels)), sharex=True)
if len(channels) == 1:
    axes = [axes]

colors = plt.cm.tab10.colors
for ax, ch, color in zip(axes, channels, colors):
    ax.plot(df.index, df[ch], linewidth=0.5, color=color, alpha=0.8)
    ax.set_ylabel(ch, fontsize=12)
    ax.grid(True, alpha=0.3)

axes[0].set_title('Временные ряды 8 каналов датасета Power Consumption of Tetouan City',
                  fontsize=11, fontweight='bold')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig7_timeseries_all_channels.png', dpi=150, bbox_inches='tight')
plt.close()
print("Рисунок 12 сохранён.")

# ─────────────────────────────────────────────────────────────────
# 3. Описательная статистика и диаграммы размаха
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2.3.3 / 2.3.5  Описательная статистика и диаграммы размаха")
print("=" * 60)

stats_df = df[channels].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
print(stats_df.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Статистика таблицей
ax = axes[0]
ax.axis('off')
table_data = [['Канал', 'Среднее', 'Std', 'Мин', 'Макс']]
for ch in channels:
    row = [ch, f"{df[ch].mean():.1f}", f"{df[ch].std():.1f}",
           f"{df[ch].min():.1f}", f"{df[ch].max():.1f}"]
    table_data.append(row)
tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               loc='center', cellLoc='center')
tbl.auto_set_font_size(True)
tbl.scale(1.2, 1.5)
ax.set_title('Описательная статистика', fontweight='bold')

# Диаграммы размаха
axes[1].boxplot([df[ch].dropna().values for ch in channels],
                labels=channels, vert=True)
axes[1].set_title('Диаграммы размаха по каналам', fontweight='bold')
axes[1].tick_params(axis='x', rotation=30)
axes[1].grid(True, alpha=0.3, axis='y')

fig.suptitle('Описательная статистика и диаграммы размаха каналов',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig8_stats_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Рисунок 13 сохранён.")

# ─────────────────────────────────────────────────────────────────
# 4. Пропущенные значения и выбросы (3σ)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2.3.4  Пропущенные значения и выбросы")
print("=" * 60)

missing = df[channels].isnull().sum()
print("Пропущенные значения:")
print(missing.to_string())

print("\nВыбросы (критерий 3σ):")
for ch in channels:
    mu, sigma = df[ch].mean(), df[ch].std()
    outliers = ((df[ch] - mu).abs() > 3 * sigma).sum()
    pct = outliers / len(df) * 100
    print(f"  {ch:20s}: {outliers:5d} ({pct:.2f}%)")

# ─────────────────────────────────────────────────────────────────
# 5. Корреляционный анализ
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2.3.6  Корреляционный анализ каналов")
print("=" * 60)

corr = df[channels].corr(method='pearson')
print("Матрица корреляций Пирсона:")
print(corr.round(3).to_string())

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, square=True,
            cbar_kws={'shrink': 0.8})
ax.set_title('Матрица корреляций Пирсона для 8 каналов',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig9_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Рисунок 14 сохранён.")

# ─────────────────────────────────────────────────────────────────
# 6. Декомпозиция и анализ шума (SNR) — Zone 1
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2.3.7  Декомпозиция и анализ шума (SNR)")
print("=" * 60)

zone1_daily = df['zone1'].resample('h').mean().dropna()  # часовой ресемпл
period = 24  # суточный цикл в часах

decomp = seasonal_decompose(zone1_daily, model='additive', period=period)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
components = [
    (zone1_daily, 'Исходный ряд (Zone 1)'),
    (decomp.trend, 'Тренд'),
    (decomp.seasonal, 'Сезонность'),
    (decomp.resid, 'Остатки'),
]
for ax, (data, label) in zip(axes, components):
    ax.plot(data.index, data.values, linewidth=0.8, color='steelblue')
    ax.set_ylabel(label, fontsize=12)
    ax.grid(True, alpha=0.3)
axes[0].set_title(
    'Декомпозиция Zone 1 Power Consumption (тренд, сезонность, остатки)',
    fontsize=11, fontweight='bold')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig10_decomposition_zone1.png', dpi=150, bbox_inches='tight')
plt.close()
print("Рисунок 15 сохранён.")

# Гистограмма остатков
resid = decomp.resid.dropna()
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(resid.values, bins=60, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel('Остатки')
ax.set_ylabel('Частота')
ax.set_title('Гистограмма остатков декомпозиции Zone 1',
             fontsize=11, fontweight='bold')
skewness = stats.skew(resid.values)
ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label=f'0 (skewness={skewness:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig11_residuals_histogram.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Рисунок 16 сохранён. Skewness остатков: {skewness:.3f}")

# SNR
signal_var = np.var(decomp.trend.dropna().values) + np.var(decomp.seasonal.dropna().values)
noise_var  = np.var(resid.values)
snr_db = 10 * np.log10(signal_var / noise_var) if noise_var > 0 else float('inf')
print(f"\nSNR = 10·log10({signal_var:.0f} / {noise_var:.0f}) ≈ {snr_db:.1f} дБ")
if snr_db >= 15:
    print("Оценка: «Отлично» — данные пригодны без дополнительной фильтрации")
elif snr_db >= 10:
    print("Оценка: «Хорошо»")
else:
    print("Оценка: «Удовлетворительно» — рекомендуется сглаживание")

# Выбросы на временном ряду Zone 1 (часовой)
mu, sigma = zone1_daily.mean(), zone1_daily.std()
is_outlier = (zone1_daily - mu).abs() > 3 * sigma
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(zone1_daily.index, zone1_daily.values, linewidth=0.6, color='steelblue', label='Zone 1')
ax.scatter(zone1_daily[is_outlier].index, zone1_daily[is_outlier].values,
           color='red', s=20, zorder=5, label=f'Выбросы ({is_outlier.sum()})')
ax.set_ylabel('Потребление, кВт')
ax.set_title('Выбросы Zone 1 по критерию 3σ (временная привязка)',
             fontsize=11, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig12_outliers_zone1.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Рисунок 17 сохранён. Выбросов: {is_outlier.sum()} ({is_outlier.mean()*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────
# 7. Вывод по разделу
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ВЫВОД ПО РАЗДЕЛУ 2")
print("=" * 60)
print(f"  Записей: {len(df):,}  Каналов: {len(channels)}")
print(f"  Период: {df.index.min().date()} — {df.index.max().date()}")
print(f"  Пропуски: {df[channels].isnull().sum().sum()}")
zone1_out = ((df['zone1'] - df['zone1'].mean()).abs() > 3*df['zone1'].std()).sum()
print(f"  Выбросы Zone1: {zone1_out} ({zone1_out/len(df)*100:.2f}%)")
print(f"  SNR Zone1: {snr_db:.1f} дБ")
print("  Вывод: данные пригодны для LSTM/GRU/Prophet без дополнительной фильтрации")
print("\nВсе рисунки сохранены в:", OUTPUT_DIR)

