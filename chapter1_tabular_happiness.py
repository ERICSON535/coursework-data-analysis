"""
Глава 1. Первичный анализ набора табличных данных
Датасет: World Happiness Report 2021
Источник: https://www.kaggle.com/datasets/mathurinache/world-happiness-report-2021

Инструкция по загрузке:
  1. Зарегистрируйтесь на kaggle.com
  2. Скачайте датасет: kaggle datasets download -d mathurinache/world-happiness-report-2021
  3. Распакуйте в папку data/ рядом с этим скриптом
  4. Файл должен называться: data/world-happiness-report-2021.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 100
OUTPUT_DIR = 'plots/chapter1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = 'data/world-happiness-report-2021.csv'

# ── 1. Загрузка данных ─────────────────────────────────────────────────────────
print("=" * 60)
print("1. ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

# Переименуем столбцы для удобства
rename_map = {
    'Country name': 'country',
    'Regional indicator': 'region',
    'Ladder score': 'happiness',
    'Logged GDP per capita': 'gdp',
    'Social support': 'social_support',
    'Healthy life expectancy': 'life_expectancy',
    'Freedom to make life choices': 'freedom',
    'Generosity': 'generosity',
    'Perceptions of corruption': 'corruption',
    'Ladder score in Dystopia': 'dystopia_ladder',
    'Explained by: Log GDP per capita': 'expl_gdp',
    'Explained by: Social support': 'expl_social',
    'Explained by: Healthy life expectancy': 'expl_life',
    'Explained by: Freedom to make life choices': 'expl_freedom',
    'Explained by: Generosity': 'expl_generosity',
    'Explained by: Perceptions of corruption': 'expl_corruption',
    'Dystopia + residual': 'dystopia_residual',
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

print(f"Размер датасета: {df.shape[0]} строк × {df.shape[1]} столбцов")
print(f"\nПервые 5 строк:")
print(df.head())
print(f"\nТипы данных:\n{df.dtypes}")

# ── 2. Анализ пропущенных значений ────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ")
print("=" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Пропусков': missing, '%': missing_pct})
missing_df = missing_df[missing_df['Пропусков'] > 0]
if missing_df.empty:
    print("Пропущенные значения отсутствуют.")
else:
    print(missing_df)

# ── 3. Описательная статистика ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. ОПИСАТЕЛЬНАЯ СТАТИСТИКА")
print("=" * 60)

num_cols = ['happiness', 'gdp', 'social_support', 'life_expectancy',
            'freedom', 'generosity', 'corruption']
num_cols = [c for c in num_cols if c in df.columns]
print(df[num_cols].describe().round(3))

# ── 4. Распределение индекса счастья ──────────────────────────────────────────
print("\n" + "=" * 60)
print("4. ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЙ")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Распределения числовых признаков\nWorld Happiness Report 2021',
             fontsize=14, fontweight='bold')

for ax, col in zip(axes.flatten(), num_cols[:6]):
    ax.hist(df[col].dropna(), bins=25, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.5, label=f'Среднее: {df[col].mean():.2f}')
    ax.axvline(df[col].median(), color='orange', linestyle='--', linewidth=1.5, label=f'Медиана: {df[col].median():.2f}')
    ax.set_title(col.replace('_', ' ').title())
    ax.set_xlabel('Значение')
    ax.set_ylabel('Количество стран')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_distributions.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/fig1_distributions.png")

# ── 5. Корреляционная матрица ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("=" * 60)

corr_matrix = df[num_cols].corr()
print("Корреляционная матрица:")
print(corr_matrix.round(3))

# Признаки, сильно коррелирующие с индексом счастья
if 'happiness' in corr_matrix.columns:
    corr_with_happiness = corr_matrix['happiness'].drop('happiness').sort_values(ascending=False)
    print(f"\nКорреляция признаков с индексом счастья:\n{corr_with_happiness.round(3)}")

# Высококоррелированные пары (r > 0.80)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.80:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], round(r, 3)))
if high_corr_pairs:
    print(f"\nВысококоррелированные пары (|r| > 0.80):")
    for a, b, r in high_corr_pairs:
        print(f"  {a} ↔ {b}: r = {r}")

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            mask=mask, ax=ax, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
ax.set_title('Корреляционная матрица\nWorld Happiness Report 2021', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_correlation.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/fig2_correlation.png")

# ── 6. Анализ по регионам ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. АНАЛИЗ ПО РЕГИОНАМ")
print("=" * 60)

if 'region' in df.columns and 'happiness' in df.columns:
    region_stats = df.groupby('region')['happiness'].agg(['mean', 'median', 'std', 'count'])
    region_stats.columns = ['Среднее', 'Медиана', 'Std', 'Кол-во стран']
    region_stats = region_stats.sort_values('Среднее', ascending=False)
    print(region_stats.round(3))

    fig, ax = plt.subplots(figsize=(13, 6))
    region_order = region_stats.index.tolist()
    df_sorted = df.copy()
    df_sorted['region'] = pd.Categorical(df_sorted['region'], categories=region_order, ordered=True)
    df_sorted = df_sorted.sort_values('region')
    bp = df_sorted.boxplot(column='happiness', by='region', ax=ax,
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='navy'),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='navy'),
                    capprops=dict(color='navy'),
                    flierprops=dict(marker='o', color='gray', alpha=0.5))
    ax.set_xticklabels(region_order, rotation=45, ha='right', fontsize=8)
    ax.set_title('Распределение индекса счастья по регионам мира', fontsize=13, fontweight='bold')
    ax.set_xlabel('Регион')
    ax.set_ylabel('Индекс счастья')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig3_regions_boxplot.png', bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {OUTPUT_DIR}/fig3_regions_boxplot.png")

# ── 7. Топ и антирейтинг стран ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. ТОП-10 И АНТИРЕЙТИНГ-10 СТРАН")
print("=" * 60)

if 'country' in df.columns and 'happiness' in df.columns:
    df_sorted = df.sort_values('happiness', ascending=False)
    top10 = df_sorted.head(10)[['country', 'happiness']]
    bot10 = df_sorted.tail(10)[['country', 'happiness']]
    print("Топ-10 самых счастливых стран:")
    print(top10.to_string(index=False))
    print("\nАнтирейтинг-10 (наименее счастливые):")
    print(bot10.to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.barh(top10['country'], top10['happiness'], color='seagreen', edgecolor='white')
    ax1.set_title('Топ-10 самых счастливых стран', fontweight='bold')
    ax1.set_xlabel('Индекс счастья')
    ax1.invert_yaxis()
    ax1.set_xlim(0, 8)

    ax2.barh(bot10['country'], bot10['happiness'], color='tomato', edgecolor='white')
    ax2.set_title('10 наименее счастливых стран', fontweight='bold')
    ax2.set_xlabel('Индекс счастья')
    ax2.invert_yaxis()
    ax2.set_xlim(0, 8)

    plt.suptitle('World Happiness Report 2021', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig4_top_countries.png', bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {OUTPUT_DIR}/fig4_top_countries.png")

# ── 8. Scatter: ВВП vs индекс счастья ────────────────────────────────────────
if 'gdp' in df.columns and 'happiness' in df.columns:
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(df['gdp'], df['happiness'], alpha=0.7,
                         c=df['life_expectancy'] if 'life_expectancy' in df.columns else 'steelblue',
                         cmap='viridis', s=60, edgecolors='none')
    if 'life_expectancy' in df.columns:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Ожидаемая продолжительность жизни', fontsize=9)

    slope, intercept, r_value, p_value, _ = stats.linregress(
        df['gdp'].dropna(), df.loc[df['gdp'].notna(), 'happiness'])
    x_line = np.linspace(df['gdp'].min(), df['gdp'].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2,
            label=f'Тренд (r={r_value:.2f})')
    ax.set_xlabel('ВВП на душу населения (log)', fontsize=11)
    ax.set_ylabel('Индекс счастья', fontsize=11)
    ax.set_title('Зависимость индекса счастья от ВВП\nWorld Happiness Report 2021',
                 fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig5_gdp_happiness.png', bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {OUTPUT_DIR}/fig5_gdp_happiness.png")

# ── 9. Обнаружение выбросов (boxplot) ────────────────────────────────────────
print("\n" + "=" * 60)
print("8. ОБНАРУЖЕНИЕ ВЫБРОСОВ (метод IQR)")
print("=" * 60)

for col in ['happiness', 'gdp', 'generosity']:
    if col not in df.columns:
        continue
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} выбросов (границы: [{lower:.3f}, {upper:.3f}])")
    if not outliers.empty and 'country' in df.columns:
        print(f"  Страны-выбросы: {', '.join(outliers['country'].values[:5])}")

fig, axes = plt.subplots(1, len(num_cols), figsize=(16, 5))
for ax, col in zip(axes, num_cols):
    ax.boxplot(df[col].dropna(), patch_artist=True,
               boxprops=dict(facecolor='lightcyan', color='steelblue'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title(col.replace('_', ' ').title(), fontsize=8)
    ax.set_xticks([])
fig.suptitle('Ящики с усами: выявление выбросов\nWorld Happiness Report 2021',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig6_outliers.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/fig6_outliers.png")

# ── 10. Seaborn PairPlot — попарные диаграммы рассеяния ──────────────────────
print("\n" + "=" * 60)
print("10. SEABORN PAIRPLOT — ПОПАРНЫЕ ДИАГРАММЫ РАССЕЯНИЯ")
print("=" * 60)

pp_cols1 = [c for c in ['happiness', 'gdp', 'social_support'] if c in df.columns]
pp_cols2 = [c for c in ['happiness', 'life_expectancy', 'freedom', 'generosity'] if c in df.columns]

# PairPlot 1: счастье, ВВП, соц. поддержка
g1 = sns.pairplot(df[pp_cols1].dropna(), diag_kind='kde', plot_kws={'alpha': 0.6, 'color': 'steelblue'})
g1.fig.suptitle('Попарные диаграммы рассеяния: счастье, ВВП, социальная поддержка',
                y=1.02, fontsize=11, fontweight='bold')
g1.fig.savefig(f'{OUTPUT_DIR}/fig7_pairplot1.png', bbox_inches='tight')
plt.close('all')
print(f"Сохранено: {OUTPUT_DIR}/fig7_pairplot1.png")

# PairPlot 2: счастье, продолж. жизни, свобода, щедрость
g2 = sns.pairplot(df[pp_cols2].dropna(), diag_kind='hist', plot_kws={'alpha': 0.6, 'color': 'darkorange'})
g2.fig.suptitle('Попарные диаграммы рассеяния: счастье, продолжительность жизни, свобода, щедрость',
                y=1.02, fontsize=11, fontweight='bold')
g2.fig.savefig(f'{OUTPUT_DIR}/fig8_pairplot2.png', bbox_inches='tight')
plt.close('all')
print(f"Сохранено: {OUTPUT_DIR}/fig8_pairplot2.png")

# PairPlot 3: с разбивкой по укрупнённым регионам (hue)
if 'region' in df.columns:
    df_pp = df[pp_cols1 + ['region']].dropna().copy()
    # Упрощаем названия регионов до коротких меток
    region_short = {
        'Western Europe': 'W.Europe',
        'North America and ANZ': 'N.America/ANZ',
        'Middle East and North Africa': 'MENA',
        'Latin America and Caribbean': 'Lat.America',
        'Central and Eastern Europe': 'E.Europe',
        'East Asia': 'E.Asia',
        'Southeast Asia': 'SE.Asia',
        'Commonwealth of Independent States': 'CIS',
        'Sub-Saharan Africa': 'Sub-Sah.Africa',
        'South Asia': 'S.Asia',
    }
    df_pp['region_short'] = df_pp['region'].map(region_short).fillna(df_pp['region'])
    g3 = sns.pairplot(df_pp[pp_cols1 + ['region_short']], hue='region_short',
                      diag_kind='kde', plot_kws={'alpha': 0.55}, height=2.2)
    g3.fig.suptitle('Попарные диаграммы рассеяния с разбивкой по регионам мира',
                    y=1.02, fontsize=11, fontweight='bold')
    g3.fig.savefig(f'{OUTPUT_DIR}/fig9_pairplot3.png', bbox_inches='tight')
    plt.close('all')
    print(f"Сохранено: {OUTPUT_DIR}/fig9_pairplot3.png")

# ── 11. Пузырьковая диаграмма (аналог Plotly bubble chart) ───────────────────
print("\n" + "=" * 60)
print("11. ПУЗЫРЬКОВАЯ ДИАГРАММА: ВВП vs СЧАСТЬЕ vs ПРОДОЛЖ. ЖИЗНИ")
print("=" * 60)

if all(c in df.columns for c in ['gdp', 'happiness', 'life_expectancy', 'region']):
    fig, ax = plt.subplots(figsize=(12, 7))
    regions = df['region'].unique()
    cmap = plt.cm.get_cmap('tab10', len(regions))
    region_color = {r: cmap(i) for i, r in enumerate(regions)}

    for _, row in df.iterrows():
        size = (row['life_expectancy'] / df['life_expectancy'].max()) ** 2 * 600 + 30
        ax.scatter(row['gdp'], row['happiness'],
                   s=size, color=region_color[row['region']],
                   alpha=0.65, edgecolors='white', linewidths=0.5)

    # Легенда по регионам
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=region_color[r],
                      markersize=8, label=r) for r in regions]
    ax.legend(handles=handles, loc='upper left', fontsize=7, ncol=2,
              framealpha=0.8, title='Регион')
    ax.set_xlabel('ВВП на душу населения (log)', fontsize=11)
    ax.set_ylabel('Индекс счастья', fontsize=11)
    ax.set_title('Пузырьковая диаграмма: ВВП, счастье, продолж. жизни (размер пузыря)\n'
                 'World Happiness Report 2021', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig10_plotly_bubble.png', bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {OUTPUT_DIR}/fig10_plotly_bubble.png")

# ── 12. Аугментация данных — добавление шума ─────────────────────────────────
print("\n" + "=" * 60)
print("12. АУГМЕНТАЦИЯ ДАННЫХ — ДОБАВЛЕНИЕ ГАУССОВСКОГО ШУМА")
print("=" * 60)

if 'happiness' in df.columns and 'gdp' in df.columns:
    np.random.seed(42)
    noise_cols = [c for c in ['happiness', 'gdp', 'social_support'] if c in df.columns]
    df_orig = df[noise_cols].dropna()
    noise_sigma = 0.05
    df_noisy = df_orig + np.random.normal(0, noise_sigma, df_orig.shape)

    fig, axes = plt.subplots(1, len(noise_cols), figsize=(14, 5))
    for ax, col in zip(axes, noise_cols):
        ax.hist(df_orig[col], bins=20, alpha=0.6, color='steelblue', label='Исходные', density=True)
        ax.hist(df_noisy[col], bins=20, alpha=0.6, color='tomato', label='С шумом', density=True)
        ax.set_title(col.replace('_', ' ').title(), fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlabel('Значение')
    fig.suptitle(f'Аугментация данных: добавление гауссовского шума (σ={noise_sigma})\n'
                 'World Happiness Report 2021', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig11_noise.png', bbox_inches='tight')
    plt.close()
    n_augmented = len(df_orig) + len(df_noisy)
    print(f"Оригинальных записей: {len(df_orig)}, с аугментацией: {n_augmented}")
    print(f"Сохранено: {OUTPUT_DIR}/fig11_noise.png")

# ── 13. Новый категориальный признак на основе непрерывного ──────────────────
print("\n" + "=" * 60)
print("13. СОЗДАНИЕ НОВОГО КАТЕГОРИАЛЬНОГО ПРИЗНАКА")
print("=" * 60)

if 'happiness' in df.columns:
    bins = [0, 4.0, 5.5, 6.5, 10]
    labels = ['Низкое', 'Ниже среднего', 'Выше среднего', 'Высокое']
    df['happiness_cat'] = pd.cut(df['happiness'], bins=bins, labels=labels)
    print("Распределение по новой категории:")
    print(df['happiness_cat'].value_counts().sort_index())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts = df['happiness_cat'].value_counts().sort_index()
    colors_cat = ['#d73027', '#fdae61', '#a6d96a', '#1a9850']
    axes[0].bar(counts.index.astype(str), counts.values, color=colors_cat, edgecolor='white')
    axes[0].set_title('Количество стран по уровню счастья', fontweight='bold')
    axes[0].set_xlabel('Уровень счастья')
    axes[0].set_ylabel('Количество стран')

    for cat, color in zip(labels, colors_cat):
        data = df[df['happiness_cat'] == cat]['happiness'].dropna()
        if not data.empty:
            axes[1].hist(data, bins=12, alpha=0.65, label=cat, color=color, density=True)
    axes[1].set_title('Распределение индекса счастья по категориям', fontweight='bold')
    axes[1].set_xlabel('Индекс счастья')
    axes[1].set_ylabel('Плотность')
    axes[1].legend(fontsize=9)

    fig.suptitle('Новый категориальный признак: уровень счастья\nWorld Happiness Report 2021',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig12_new_category.png', bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {OUTPUT_DIR}/fig12_new_category.png")

# ── 14. Вывод ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ИТОГОВЫЙ ВЫВОД")
print("=" * 60)
print(f"Датасет содержит {df.shape[0]} наблюдений и {df.shape[1]} признаков.")
print(f"Пропущенные значения: {'отсутствуют' if missing.sum() == 0 else f'{missing.sum()} значений'}.")
if 'happiness' in df.columns:
    print(f"Индекс счастья: min={df['happiness'].min():.2f}, "
          f"max={df['happiness'].max():.2f}, mean={df['happiness'].mean():.2f}.")
print("Наиболее сильная корреляция с индексом счастья — у ВВП на душу населения.")
print("Датасет пригоден для задач регрессии и анализа влияния факторов на благополучие населения.")
print(f"\nВсе графики сохранены в директории: {OUTPUT_DIR}/")
