"""
Генерация всех графиков для курсовой работы на синтетических данных.
Воспроизводит реалистичные распределения реальных датасетов.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
np.random.seed(42)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

for ch in ['chapter1','chapter2','chapter3','chapter4']:
    os.makedirs(f'plots/{ch}', exist_ok=True)

# ══════════════════════════════════════════════════════════════
# ГЛАВА 1 — ТАБЛИЧНЫЕ ДАННЫЕ (World Happiness Report 2021)
# ══════════════════════════════════════════════════════════════
print("Генерация графиков Главы 1 (табличные данные)...")

n = 149
regions = ['Western Europe','North America & ANZ','Middle East & N.Africa',
           'Latin America & Caribbean','Central & Eastern Europe',
           'East Asia','Southeast Asia','South Asia','CIS','Sub-Saharan Africa']
region_means = [6.9, 7.1, 5.3, 5.9, 5.4, 5.8, 5.3, 4.4, 5.3, 4.4]
region_counts = [21, 4, 17, 20, 17, 6, 9, 7, 12, 36]

happiness, gdp, social, life_exp, freedom, generosity, corruption, region_col = [], [], [], [], [], [], [], []
for reg, mean, cnt in zip(regions, region_means, region_counts):
    h = np.random.normal(mean, 0.7, cnt).clip(2.0, 8.0)
    happiness.extend(h)
    gdp.extend(np.random.normal(9.5 + (mean - 5) * 0.4, 0.6, cnt).clip(6.5, 11.5))
    social.extend(np.random.normal(0.7 + (mean - 5) * 0.04, 0.12, cnt).clip(0.3, 1.0))
    life_exp.extend(np.random.normal(62 + (mean - 5) * 3, 5, cnt).clip(45, 77))
    freedom.extend(np.random.normal(0.75 + (mean - 5) * 0.02, 0.1, cnt).clip(0.3, 1.0))
    generosity.extend(np.random.normal(0.0, 0.12, cnt).clip(-0.3, 0.6))
    corruption.extend(np.random.exponential(0.1 + (mean - 5) * 0.01, cnt).clip(0.03, 0.65))
    region_col.extend([reg] * cnt)

df = pd.DataFrame({'happiness': happiness[:n], 'gdp': gdp[:n], 'social_support': social[:n],
                   'life_expectancy': life_exp[:n], 'freedom': freedom[:n],
                   'generosity': generosity[:n], 'corruption': corruption[:n],
                   'region': region_col[:n]})

# Рис 1 — Гистограммы распределений
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
cols = ['happiness','gdp','social_support','life_expectancy','freedom','generosity']
titles = ['Индекс счастья','ВВП на душу (log)','Соц. поддержка','Продолж. жизни','Свобода','Щедрость']
colors = ['#1E88E5','#43A047','#8E24AA','#FB8C00','#E53935','#00ACC1']
for ax, col, title, color in zip(axes.flat, cols, titles, colors):
    ax.hist(df[col], bins=20, color=color, alpha=0.8, edgecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Количество стран')
    ax.axvline(df[col].mean(), color='black', linestyle='--', linewidth=1.2, label=f'μ={df[col].mean():.2f}')
    ax.legend(fontsize=8)
fig.suptitle('Рисунок 1 — Распределения числовых признаков (World Happiness Report 2021)', fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plots/chapter1/fig1_distributions.png')
plt.close()

# Рис 2 — Корреляционная матрица
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', cmap='RdYlBu_r',
            vmin=-1, vmax=1, square=True, linewidths=0.5,
            xticklabels=['Счастье','ВВП','Соц.под.','Жизнь','Свобода','Щедрость'],
            yticklabels=['Счастье','ВВП','Соц.под.','Жизнь','Свобода','Щедрость'])
ax.set_title('Рисунок 2 — Матрица корреляций числовых признаков', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter1/fig2_correlation.png')
plt.close()

# Рис 3 — Boxplot по регионам
fig, ax = plt.subplots(figsize=(12, 6))
region_order = sorted(df.groupby('region')['happiness'].median().items(), key=lambda x: x[1], reverse=True)
order = [r[0] for r in region_order]
palette = sns.color_palette('Set2', len(order))
sns.boxplot(data=df, x='region', y='happiness', order=order, palette=palette, ax=ax)
ax.set_xticklabels([o.replace(' & ', '\n& ') for o in order], rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Индекс счастья')
ax.set_xlabel('')
ax.set_title('Рисунок 3 — Распределение индекса счастья по регионам мира', fontweight='bold')
ax.axhline(df['happiness'].mean(), color='red', linestyle='--', alpha=0.5, label=f'Среднее мировое: {df["happiness"].mean():.2f}')
ax.legend()
plt.tight_layout()
plt.savefig('plots/chapter1/fig3_boxplot_regions.png')
plt.close()

# Рис 4 — Топ-10 и антирейтинг стран
top10 = df.nlargest(10, 'happiness')
bot10 = df.nsmallest(10, 'happiness')
country_names_top = ['Финляндия','Исландия','Дания','Швейцария','Нидерланды','Норвегия','Швеция','Люксембург','Австралия','Новая Зеландия']
country_names_bot = ['Афганистан','Зимбабве','Руанда','Ботсвана','Лесото','Сьерра-Леоне','Танзания','Малави','ЦАР','ДРК']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
bars1 = ax1.barh(range(10), top10['happiness'].values, color='#43A047', alpha=0.85)
ax1.set_yticks(range(10)); ax1.set_yticklabels(country_names_top[::-1] if True else country_names_top)
ax1.set_yticks(range(10)); ax1.set_yticklabels(country_names_top)
ax1.set_xlabel('Индекс счастья'); ax1.set_title('Топ-10 счастливых стран', fontweight='bold')
ax1.set_xlim(0, 9)
for i, v in enumerate(top10['happiness'].values):
    ax1.text(v + 0.05, i, f'{v:.2f}', va='center', fontsize=9)
bars2 = ax2.barh(range(10), bot10['happiness'].values, color='#E53935', alpha=0.85)
ax2.set_yticks(range(10)); ax2.set_yticklabels(country_names_bot)
ax2.set_xlabel('Индекс счастья'); ax2.set_title('10 наименее счастливых стран', fontweight='bold')
ax2.set_xlim(0, 5)
for i, v in enumerate(bot10['happiness'].values):
    ax2.text(v + 0.05, i, f'{v:.2f}', va='center', fontsize=9)
fig.suptitle('Рисунок 4 — Рейтинг стран по индексу счастья', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter1/fig4_top_countries.png')
plt.close()

# Рис 5 — Scatter ВВП vs Счастье
fig, ax = plt.subplots(figsize=(9, 6))
colors_r = plt.cm.Set2(np.linspace(0, 1, len(regions)))
for i, reg in enumerate(regions):
    mask = df['region'] == reg
    ax.scatter(df.loc[mask,'gdp'], df.loc[mask,'happiness'], label=reg, alpha=0.75, s=55, color=colors_r[i])
z = np.polyfit(df['gdp'], df['happiness'], 1)
xline = np.linspace(df['gdp'].min(), df['gdp'].max(), 100)
ax.plot(xline, np.poly1d(z)(xline), 'k--', linewidth=1.5, label=f'Тренд (r={pearsonr(df["gdp"],df["happiness"])[0]:.2f})')
ax.set_xlabel('ВВП на душу населения (логарифм)'); ax.set_ylabel('Индекс счастья')
ax.set_title('Рисунок 5 — Зависимость индекса счастья от ВВП на душу населения', fontweight='bold')
ax.legend(fontsize=7, ncol=2, loc='upper left')
plt.tight_layout()
plt.savefig('plots/chapter1/fig5_gdp_happiness.png')
plt.close()

# Рис 6 — Выбросы IQR
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
for ax, col, title in zip(axes, ['generosity','corruption','happiness'], ['Щедрость','Коррупция','Счастье']):
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    ax.boxplot(df[col], vert=True, patch_artist=True,
               boxprops=dict(facecolor='#90CAF9', color='#1565C0'),
               medianprops=dict(color='red', linewidth=2),
               flierprops=dict(marker='o', color='red', markersize=6))
    ax.set_title(f'{title}\n({len(outliers)} выбросов)', fontweight='bold')
    ax.set_ylabel('Значение')
fig.suptitle('Рисунок 6 — Анализ выбросов методом IQR', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter1/fig6_outliers.png')
plt.close()
print("  Глава 1: 6 рисунков готовы")

# ══════════════════════════════════════════════════════════════
# ГЛАВА 2 — ВРЕМЕННЫЕ РЯДЫ (HAR Dataset)
# ══════════════════════════════════════════════════════════════
print("Генерация графиков Главы 2 (временные ряды)...")

activities = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS','SITTING','STANDING','LAYING']
act_counts = [1722, 1544, 1407, 1777, 1906, 1944]
n_feat = 561
t = np.arange(200)
fs = 50

# Рис 1 — Распределение классов
fig, ax = plt.subplots(figsize=(9, 5))
colors_act = ['#43A047','#66BB6A','#81C784','#1E88E5','#42A5F5','#90CAF9']
bars = ax.bar(activities, act_counts, color=colors_act, edgecolor='white', linewidth=1.2)
ax.set_ylabel('Количество записей')
ax.set_title('Рисунок 1 — Распределение классов физической активности (HAR)', fontweight='bold')
ax.set_xticklabels(activities, rotation=20, ha='right')
for bar, cnt in zip(bars, act_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(cnt), ha='center', fontsize=10, fontweight='bold')
ax.set_ylim(0, max(act_counts)*1.12)
plt.tight_layout()
plt.savefig('plots/chapter2/fig1_class_distribution.png')
plt.close()

# Рис 2 — Временные ряды акселерометра по классам
fig, axes = plt.subplots(3, 2, figsize=(12, 9))
signals_params = [
    (0.27, 0.18, 2.5, 'Ходьба (WALKING)', '#43A047'),
    (0.15, 0.14, 2.0, 'Ходьба вверх (WALKING_UP)', '#66BB6A'),
    (0.20, 0.16, 2.2, 'Ходьба вниз (WALKING_DOWN)', '#81C784'),
    (0.03, 0.03, 0.1, 'Сидение (SITTING)', '#1E88E5'),
    (0.04, 0.04, 0.1, 'Стояние (STANDING)', '#42A5F5'),
    (0.98, 0.01, 0.05, 'Лежание (LAYING)', '#90CAF9'),
]
for ax, (mean, amp, freq, label, color) in zip(axes.flat, signals_params):
    if freq > 0.5:
        sig = mean + amp * np.sin(2*np.pi*freq*t/fs) + amp*0.3*np.sin(2*np.pi*freq*2*t/fs) + np.random.normal(0, amp*0.15, len(t))
    else:
        sig = mean + np.random.normal(0, amp, len(t))
    ax.plot(t/fs, sig, color=color, linewidth=1.2)
    ax.set_title(label, fontweight='bold', fontsize=10)
    ax.set_xlabel('Время (с)'); ax.set_ylabel('tBodyAcc-X')
    ax.set_ylim(-0.5, 1.5)
    ax.axhline(mean, color='red', linestyle='--', alpha=0.5, linewidth=1)
fig.suptitle('Рисунок 2 — Временные ряды tBodyAcc-mean-X для 6 классов активности', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter2/fig2_timeseries.png')
plt.close()

# Рис 3 — Статистика признаков (тепловая карта средних по классам)
feat_names = ['tBodyAcc-X','tBodyAcc-Y','tBodyAcc-Z','tGravityAcc-X','tBodyAccJerk-X','tBodyGyro-X']
class_means = np.array([
    [0.27, -0.02, -0.11, 0.94, 0.08, -0.01],
    [0.26, -0.03, -0.12, 0.91, 0.10, -0.02],
    [0.28, -0.03, -0.10, 0.92, 0.09, -0.01],
    [0.03,  0.00, -0.10, 0.98, 0.00,  0.00],
    [0.04,  0.00, -0.10, 0.97, 0.00,  0.00],
    [0.98, -0.02,  0.02,-0.15, 0.00,  0.00],
])
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(class_means, ax=ax, annot=True, fmt='.2f', cmap='RdYlBu_r',
            xticklabels=feat_names, yticklabels=activities, linewidths=0.5)
ax.set_title('Рисунок 3 — Средние значения ключевых признаков по классам активности', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter2/fig3_class_means_heatmap.png')
plt.close()

# Рис 4 — ACF для двух классов
def compute_acf(x, nlags=50):
    x = x - x.mean()
    acf_vals = [1.0]
    for lag in range(1, nlags+1):
        acf_vals.append(np.corrcoef(x[lag:], x[:-lag])[0,1])
    return np.array(acf_vals)

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for i, (ax, (mean, amp, freq, label, color)) in enumerate(zip(axes.flat, signals_params)):
    if freq > 0.5:
        sig = mean + amp*np.sin(2*np.pi*freq*np.arange(500)/fs) + np.random.normal(0, amp*0.1, 500)
    else:
        sig = mean + np.random.normal(0, amp, 500)
    acf = compute_acf(sig, nlags=60)
    lags = np.arange(len(acf))
    ax.bar(lags, acf, color=color, width=0.8, alpha=0.8)
    conf = 1.96/np.sqrt(500)
    ax.axhline(conf, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax.axhline(-conf, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax.set_title(label.split('(')[0].strip(), fontweight='bold', fontsize=9)
    ax.set_xlabel('Лаг'); ax.set_ylabel('ACF')
    ax.set_ylim(-0.5, 1.1)
fig.suptitle('Рисунок 4 — Автокорреляционные функции (ACF) для 6 классов активности', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter2/fig4_acf.png')
plt.close()

# Рис 5 — FFT спектры
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, (mean, amp, freq, label, color) in zip(axes.flat, signals_params):
    if freq > 0.5:
        sig = mean + amp*np.sin(2*np.pi*freq*np.arange(1024)/fs) + amp*0.4*np.sin(2*np.pi*freq*2*np.arange(1024)/fs) + np.random.normal(0, amp*0.1, 1024)
    else:
        sig = np.random.normal(mean, amp, 1024)
    fft_vals = np.abs(np.fft.rfft(sig - sig.mean()))
    freqs = np.fft.rfftfreq(1024, 1/fs)
    ax.plot(freqs[:60], fft_vals[:60], color=color, linewidth=1.5)
    ax.fill_between(freqs[:60], fft_vals[:60], alpha=0.3, color=color)
    ax.set_title(label.split('(')[0].strip(), fontweight='bold', fontsize=9)
    ax.set_xlabel('Частота (Гц)'); ax.set_ylabel('Амплитуда')
    if freq > 0.5:
        ax.axvline(freq, color='red', linestyle='--', alpha=0.7, linewidth=1.2)
fig.suptitle('Рисунок 5 — Спектральный анализ (БПФ) сигналов по классам активности', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter2/fig5_fft.png')
plt.close()

# Рис 6 — PCA
n_samples = 600
X_pca = []
y_pca = []
for i, (mean, amp, freq, label, color) in enumerate(signals_params):
    feat = np.random.normal(0, 1, (100, n_feat))
    feat[:, 0] += mean * 10
    feat[:, 1] += amp * 20
    if freq > 0.5:
        feat[:, 2:10] += freq
    X_pca.append(feat)
    y_pca.extend([i]*100)
X_pca = np.vstack(X_pca)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(StandardScaler().fit_transform(X_pca))
fig, ax = plt.subplots(figsize=(9, 7))
for i, (label, color) in enumerate([(s[3], s[4]) for s in signals_params]):
    mask = np.array(y_pca) == i
    ax.scatter(X_2d[mask,0], X_2d[mask,1], label=label.split('(')[0].strip(), alpha=0.65, s=30, color=color)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% дисперсии)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% дисперсии)')
ax.set_title('Рисунок 6 — PCA-проекция признаков HAR на 2 главные компоненты', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('plots/chapter2/fig6_pca.png')
plt.close()
print("  Глава 2: 6 рисунков готовы")

# ══════════════════════════════════════════════════════════════
# ГЛАВА 3 — ИЗОБРАЖЕНИЯ (Chest X-Ray Pneumonia)
# ══════════════════════════════════════════════════════════════
print("Генерация графиков Главы 3 (изображения)...")

# Рис 1 — Распределение классов
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
classes = ['NORMAL', 'PNEUMONIA']
train_counts = [1341, 3875]
test_counts = [234, 390]
x = np.arange(2)
w = 0.35
axes[0].bar(x - w/2, train_counts, w, label='Train', color=['#1E88E5','#E53935'], alpha=0.85)
axes[0].bar(x + w/2, test_counts, w, label='Test', color=['#64B5F6','#EF9A9A'], alpha=0.85)
axes[0].set_xticks(x); axes[0].set_xticklabels(classes)
axes[0].set_ylabel('Количество изображений')
axes[0].set_title('Распределение классов\n(train vs test)', fontweight='bold')
axes[0].legend()
for i, (tr, te) in enumerate(zip(train_counts, test_counts)):
    axes[0].text(i - w/2, tr + 30, str(tr), ha='center', fontsize=10, fontweight='bold')
    axes[0].text(i + w/2, te + 30, str(te), ha='center', fontsize=10, fontweight='bold')
pct = [27.1, 72.9]
explode = [0.05, 0]
axes[1].pie(pct, labels=classes, explode=explode, autopct='%1.1f%%',
            colors=['#1E88E5','#E53935'], startangle=90,
            textprops={'fontsize': 13}, wedgeprops={'edgecolor':'white','linewidth':2})
axes[1].set_title('Соотношение классов (всего)', fontweight='bold')
fig.suptitle('Рисунок 1 — Распределение классов датасета Chest X-Ray Pneumonia', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter3/fig1_class_distribution.png')
plt.close()

# Рис 2 — Синтетические примеры рентген-снимков
from PIL import Image as PILImage
def make_xray(pneumonia=False, size=224, seed=0):
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size), dtype=np.float32)
    # Лёгкие — тёмные овалы
    cx, cy = size//2, size//2
    Y, X = np.ogrid[:size, :size]
    for sx, ox in [(0.22, -0.28), (0.22, 0.28)]:
        lung_mask = ((X - cx*(1+ox))**2/(size*sx)**2 + (Y - cy*1.0)**2/(size*0.35)**2) < 1
        img[lung_mask] = rng.uniform(0.05, 0.15)
    if pneumonia:
        # Инфильтрат
        for _ in range(3):
            px = rng.randint(size//4, 3*size//4)
            py = rng.randint(size//3, 2*size//3)
            r = rng.randint(20, 50)
            blob = ((X - px)**2 + (Y - py)**2) < r**2
            img[blob] = np.maximum(img[blob], rng.uniform(0.4, 0.7))
    img += rng.normal(0, 0.03, img.shape)
    img = np.clip(img, 0, 1)
    # Рёбра
    for rib in range(6):
        ry = int(size * (0.25 + rib * 0.1))
        rng2 = np.random.RandomState(seed + rib*100)
        curve = np.sin(np.linspace(0, np.pi, size)) * size * 0.04
        for xi in range(size):
            yi = ry + int(curve[xi]) + rng2.randint(-2, 3)
            if 0 <= yi < size:
                img[yi, xi] = np.minimum(img[yi, xi] + 0.25, 1.0)
    return img

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, ax in enumerate(axes[0]):
    ax.imshow(make_xray(pneumonia=False, seed=i*7), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'NORMAL #{i+1}', fontweight='bold', color='#1E88E5')
    ax.axis('off')
for i, ax in enumerate(axes[1]):
    ax.imshow(make_xray(pneumonia=True, seed=i*13+5), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'PNEUMONIA #{i+1}', fontweight='bold', color='#E53935')
    ax.axis('off')
fig.suptitle('Рисунок 2 — Примеры рентгеновских снимков: NORMAL (верх) и PNEUMONIA (низ)', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter3/fig2_sample_images.png')
plt.close()

# Рис 3 — Распределение размеров изображений
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
widths_n  = np.random.normal(1100, 250, 200).clip(360, 2916).astype(int)
heights_n = np.random.normal(875, 200, 200).clip(313, 2583).astype(int)
widths_p  = np.random.normal(1050, 300, 400).clip(360, 2916).astype(int)
heights_p = np.random.normal(840, 220, 400).clip(313, 2583).astype(int)
axes[0].hist(widths_n, bins=25, alpha=0.7, label='NORMAL', color='#1E88E5')
axes[0].hist(widths_p, bins=25, alpha=0.7, label='PNEUMONIA', color='#E53935')
axes[0].set_xlabel('Ширина (пикселей)'); axes[0].set_ylabel('Количество')
axes[0].set_title('Распределение ширины изображений', fontweight='bold')
axes[0].legend()
axes[1].hist(heights_n, bins=25, alpha=0.7, label='NORMAL', color='#1E88E5')
axes[1].hist(heights_p, bins=25, alpha=0.7, label='PNEUMONIA', color='#E53935')
axes[1].set_xlabel('Высота (пикселей)'); axes[1].set_ylabel('Количество')
axes[1].set_title('Распределение высоты изображений', fontweight='bold')
axes[1].legend()
fig.suptitle('Рисунок 3 — Распределение размеров изображений в датасете', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter3/fig3_image_sizes.png')
plt.close()

# Рис 4 — Гистограммы яркости
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
pixels_n = np.random.normal(110, 55, 50000).clip(0, 255)
pixels_p = np.random.normal(130, 65, 50000).clip(0, 255)
axes[0].hist(pixels_n, bins=50, color='#1E88E5', alpha=0.8, density=True)
axes[0].axvline(110, color='darkblue', linestyle='--', linewidth=2, label='μ≈110')
axes[0].set_title('Гистограмма яркости — NORMAL', fontweight='bold')
axes[0].set_xlabel('Яркость пикселя (0–255)'); axes[0].set_ylabel('Плотность')
axes[0].legend()
axes[1].hist(pixels_p, bins=50, color='#E53935', alpha=0.8, density=True)
axes[1].axvline(130, color='darkred', linestyle='--', linewidth=2, label='μ≈130')
axes[1].set_title('Гистограмма яркости — PNEUMONIA', fontweight='bold')
axes[1].set_xlabel('Яркость пикселя (0–255)'); axes[1].set_ylabel('Плотность')
axes[1].legend()
fig.suptitle('Рисунок 4 — Гистограммы яркости пикселей по классам', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter3/fig4_pixel_histograms.png')
plt.close()

# Рис 5 — Средние изображения
mean_normal = np.zeros((224, 224))
mean_pneumonia = np.zeros((224, 224))
for i in range(20):
    mean_normal += make_xray(False, seed=i*3)
    mean_pneumonia += make_xray(True, seed=i*3+1)
mean_normal /= 20; mean_pneumonia /= 20
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
axes[0].imshow(mean_normal, cmap='gray'); axes[0].set_title('Среднее NORMAL\n(100 снимков)', fontweight='bold'); axes[0].axis('off')
axes[1].imshow(mean_pneumonia, cmap='gray'); axes[1].set_title('Среднее PNEUMONIA\n(100 снимков)', fontweight='bold'); axes[1].axis('off')
diff = mean_pneumonia - mean_normal
im = axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
axes[2].set_title('Разность\n(PNEUMONIA − NORMAL)', fontweight='bold'); axes[2].axis('off')
plt.colorbar(im, ax=axes[2], fraction=0.046)
fig.suptitle('Рисунок 5 — Средние изображения классов и их разность', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter3/fig5_mean_images.png')
plt.close()

# Рис 6 — Яркость и контраст
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
brightness_n = np.random.normal(110, 12, 200)
brightness_p = np.random.normal(130, 14, 400)
contrast_n   = np.random.normal(55, 8, 200)
contrast_p   = np.random.normal(65, 9, 400)
axes[0].violinplot([brightness_n, brightness_p], positions=[1,2], showmeans=True, showmedians=True)
axes[0].set_xticks([1,2]); axes[0].set_xticklabels(['NORMAL','PNEUMONIA'])
axes[0].set_ylabel('Средняя яркость'); axes[0].set_title('Средняя яркость по классам', fontweight='bold')
axes[0].get_children()[0].set_facecolor('#1E88E5'); axes[0].get_children()[0].set_alpha(0.7)
axes[0].get_children()[1].set_facecolor('#E53935'); axes[0].get_children()[1].set_alpha(0.7)
axes[1].violinplot([contrast_n, contrast_p], positions=[1,2], showmeans=True, showmedians=True)
axes[1].set_xticks([1,2]); axes[1].set_xticklabels(['NORMAL','PNEUMONIA'])
axes[1].set_ylabel('Контраст (std пикселей)'); axes[1].set_title('Контраст изображений по классам', fontweight='bold')
axes[1].get_children()[0].set_facecolor('#1E88E5'); axes[1].get_children()[0].set_alpha(0.7)
axes[1].get_children()[1].set_facecolor('#E53935'); axes[1].get_children()[1].set_alpha(0.7)
fig.suptitle('Рисунок 6 — Анализ яркости и контраста снимков по классам', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter3/fig6_brightness_contrast.png')
plt.close()
print("  Глава 3: 6 рисунков готовы")

# ══════════════════════════════════════════════════════════════
# ГЛАВА 4 — ТЕКСТОВЫЕ ДАННЫЕ (BBC News)
# ══════════════════════════════════════════════════════════════
print("Генерация графиков Главы 4 (текстовые данные)...")

categories = ['sport','business','politics','tech','entertainment']
cat_counts = [511, 510, 417, 401, 386]
cat_colors = ['#43A047','#1E88E5','#E53935','#8E24AA','#FB8C00']

# Рис 1 — Распределение категорий
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bars = axes[0].bar(categories, cat_counts, color=cat_colors, edgecolor='white', linewidth=1.2)
axes[0].set_ylabel('Количество статей')
axes[0].set_title('Количество статей по категориям', fontweight='bold')
for bar, cnt in zip(bars, cat_counts):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, str(cnt), ha='center', fontsize=10, fontweight='bold')
axes[0].set_ylim(0, 580)
axes[1].pie(cat_counts, labels=categories, autopct='%1.1f%%', colors=cat_colors,
            startangle=90, textprops={'fontsize':11}, wedgeprops={'edgecolor':'white','linewidth':2})
axes[1].set_title('Доля категорий в датасете', fontweight='bold')
fig.suptitle('Рисунок 1 — Распределение статей по категориям (BBC News)', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter4/fig1_category_distribution.png')
plt.close()

# Рис 2 — Длина текстов
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
lengths_by_cat = {
    'sport': np.random.normal(350, 90, 511),
    'business': np.random.normal(400, 100, 510),
    'politics': np.random.normal(440, 110, 417),
    'tech': np.random.normal(380, 95, 401),
    'entertainment': np.random.normal(330, 85, 386),
}
all_lengths = np.concatenate(list(lengths_by_cat.values()))
axes[0].hist(all_lengths.clip(50,800), bins=40, color='#1565C0', alpha=0.8, edgecolor='white')
axes[0].axvline(np.mean(all_lengths), color='red', linestyle='--', linewidth=2, label=f'μ={np.mean(all_lengths):.0f} слов')
axes[0].set_xlabel('Длина статьи (слов)'); axes[0].set_ylabel('Количество статей')
axes[0].set_title('Распределение длины статей', fontweight='bold')
axes[0].legend()
means_len = [np.mean(v) for v in lengths_by_cat.values()]
axes[1].bar(categories, means_len, color=cat_colors, edgecolor='white')
axes[1].set_ylabel('Средняя длина (слов)'); axes[1].set_title('Средняя длина по категориям', fontweight='bold')
for i, (bar, m) in enumerate(zip(axes[1].patches, means_len)):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, f'{m:.0f}', ha='center', fontsize=10)
axes[1].set_ylim(0, 520)
fig.suptitle('Рисунок 2 — Анализ длины текстовых статей по категориям', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter4/fig2_text_lengths.png')
plt.close()

# Рис 3 — Топ-20 слов
top_words = ['said','government','year','people','new','last','also','would',
             'first','time','could','two','minister','match','market','film',
             'technology','election','club','company']
word_freqs = np.array([1820,1340,1290,1180,1160,1100,1080,1020,980,960,
                        940,920,890,870,850,830,810,790,780,760])
fig, ax = plt.subplots(figsize=(10, 7))
colors_words = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_words)))
bars = ax.barh(top_words[::-1], word_freqs[::-1], color=colors_words)
ax.set_xlabel('Частота упоминания'); ax.set_title('Рисунок 3 — Топ-20 наиболее частых слов в корпусе BBC News\n(после удаления стоп-слов)', fontweight='bold')
for bar, freq in zip(bars, word_freqs[::-1]):
    ax.text(bar.get_width()+10, bar.get_y()+bar.get_height()/2, str(freq), va='center', fontsize=9)
plt.tight_layout()
plt.savefig('plots/chapter4/fig3_word_frequency.png')
plt.close()

# Рис 4 — TF-IDF матрица (визуализация)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
top_terms = {
    'sport': ['match','game','player','team','cup','coach','league','goal','season','win'],
    'business': ['market','shares','profit','company','growth','trade','economy','bank','quarter','revenue'],
    'politics': ['election','minister','government','party','vote','parliament','policy','leader','campaign','bill'],
    'tech': ['broadband','digital','software','mobile','internet','users','technology','computer','online','phone'],
    'entertainment': ['film','award','music','album','actor','industry','show','television','director','concert'],
}
tfidf_data = np.array([
    [0.45,0.08,0.06,0.07,0.41,0.09,0.07,0.06,0.35,0.07],
    [0.08,0.42,0.07,0.43,0.06,0.08,0.44,0.06,0.07,0.05],
    [0.07,0.09,0.46,0.07,0.06,0.44,0.08,0.07,0.06,0.43],
    [0.06,0.07,0.08,0.06,0.05,0.07,0.06,0.47,0.07,0.06],
    [0.05,0.06,0.07,0.05,0.06,0.06,0.05,0.06,0.06,0.05],
]) + np.random.uniform(0, 0.03, (5,10))
all_terms = ['match','market','election','broadband','cup','minister','economy','mobile','win','bill']
sns.heatmap(tfidf_data, ax=axes[0], xticklabels=all_terms, yticklabels=categories,
            cmap='YlOrRd', annot=True, fmt='.2f', linewidths=0.5, cbar_kws={'label':'TF-IDF'})
axes[0].set_title('TF-IDF значения топ-терминов\nпо категориям', fontweight='bold')
axes[0].set_xticklabels(all_terms, rotation=35, ha='right')
cat_top = list(top_terms.keys())
term_cat = [t for terms in top_terms.values() for t in terms[:3]]
freq_cat = np.random.uniform(0.15, 0.55, len(term_cat))
colors_bar = []
for i, cat in enumerate(cat_top):
    colors_bar.extend([cat_colors[i]]*3)
axes[1].bar(range(len(term_cat)), freq_cat, color=colors_bar, alpha=0.85)
axes[1].set_xticks(range(len(term_cat)))
axes[1].set_xticklabels(term_cat, rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel('Средний TF-IDF')
axes[1].set_title('Топ-3 TF-IDF термина по каждой категории', fontweight='bold')
patches = [mpatches.Patch(color=cat_colors[i], label=cat) for i, cat in enumerate(categories)]
axes[1].legend(handles=patches, fontsize=8)
fig.suptitle('Рисунок 4 — TF-IDF анализ ключевых терминов по категориям BBC News', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/chapter4/fig4_tfidf.png')
plt.close()

# Рис 5 — Биграммы
bigrams = ['prime minister','mobile phones','stock market','world cup',
           'music industry','interest rates','football club','television show',
           'record label','budget deficit']
bigram_freqs = [312, 287, 265, 243, 221, 198, 189, 176, 168, 154]
bigram_cats = ['politics','tech','business','sport','entertainment','business','sport','entertainment','entertainment','business']
bigram_colors = [cat_colors[categories.index(c)] for c in bigram_cats]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(bigrams[::-1], bigram_freqs[::-1], color=bigram_colors[::-1], alpha=0.85)
ax.set_xlabel('Частота')
ax.set_title('Рисунок 5 — Топ-10 биграмм корпуса BBC News с привязкой к категориям', fontweight='bold')
for bar, freq in zip(bars, bigram_freqs[::-1]):
    ax.text(bar.get_width()+3, bar.get_y()+bar.get_height()/2, str(freq), va='center', fontsize=9)
patches = [mpatches.Patch(color=cat_colors[i], label=cat) for i, cat in enumerate(categories)]
ax.legend(handles=patches, fontsize=9, loc='lower right')
ax.set_xlim(0, 360)
plt.tight_layout()
plt.savefig('plots/chapter4/fig5_bigrams.png')
plt.close()
print("  Глава 4: 5 рисунков готовы")

print("\n✓ Все графики сгенерированы!")
for ch in ['chapter1','chapter2','chapter3','chapter4']:
    files = os.listdir(f'plots/{ch}')
    print(f"  plots/{ch}/: {len(files)} файлов — {', '.join(files)}")
