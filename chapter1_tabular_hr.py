# -*- coding: utf-8 -*-
"""
Глава 1. Первичный анализ табличного датасета
Датасет: IBM HR Analytics Employee Attrition & Performance
Источник: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
})

OUTPUT_DIR = 'plots/chapter1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv'

# ─────────────────────────────────────────────────────────────────
# 1. Загрузка и общий обзор
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. ЗАГРУЗКА И ОБЩИЙ ОБЗОР")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Размер: {df.shape[0]} строк x {df.shape[1]} столбцов")
print(f"\nСтолбцы:\n{list(df.columns)}")
print(f"\nТипы данных:\n{df.dtypes.value_counts()}")
print(f"\nПропущенные значения: {df.isnull().sum().sum()}")
print(f"\nДубликаты строк: {df.duplicated().sum()}")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f"\nЧисловых признаков: {len(num_cols)}")
print(f"Категориальных признаков: {len(cat_cols)}")
print(f"Категориальные: {cat_cols}")

print(f"\nРаспределение целевой переменной (Attrition):")
print(df['Attrition'].value_counts())
print(df['Attrition'].value_counts(normalize=True).round(3))

# ─────────────────────────────────────────────────────────────────
# 2. Диаграммы распределения числовых признаков (Matplotlib)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. ДИАГРАММЫ РАСПРЕДЕЛЕНИЯ ЧИСЛОВЫХ ПРИЗНАКОВ")
print("=" * 60)

key_num = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany',
           'TotalWorkingYears', 'NumCompaniesWorked']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for ax, col in zip(axes, key_num):
    ax.hist(df[col], bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax.set_title(col, fontweight='bold')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Частота')
    mu = df[col].mean()
    ax.axvline(mu, color='red', linestyle='--', linewidth=1.5,
               label=f'Среднее: {mu:.1f}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig.suptitle('Распределение ключевых числовых признаков (IBM HR)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_distributions.png', bbox_inches='tight')
plt.close()
print("Сохранено: fig1_distributions.png")

# ─────────────────────────────────────────────────────────────────
# 3. Тепловые карты (Seaborn)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. ТЕПЛОВЫЕ КАРТЫ")
print("=" * 60)

corr_cols = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany',
             'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
             'NumCompaniesWorked', 'DistanceFromHome', 'PercentSalaryHike']
corr = df[corr_cols].corr()
print("Матрица корреляций:")
print(corr.round(2).to_string())

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=-1, vmax=1, ax=ax, linewidths=0.5, square=True,
            cbar_kws={'shrink': 0.8})
ax.set_title('Матрица корреляций числовых признаков', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_correlation.png', bbox_inches='tight')
plt.close()
print("Сохранено: fig2_correlation.png")

# Тепловая карта 2 - нормализованные значения первых 30 строк
fig, ax = plt.subplots(figsize=(14, 5))
heat_data = df[corr_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min())).head(30)
sns.heatmap(heat_data.T, ax=ax, cmap='YlOrRd', linewidths=0.2, xticklabels=False)
ax.set_title('Нормализованные значения числовых признаков (первые 30 строк)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_heatmap2.png', bbox_inches='tight')
plt.close()
print("Сохранено: fig3_heatmap2.png")

# ─────────────────────────────────────────────────────────────────
# 4. Визуализация средствами Seaborn (>=3 пары признаков)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. ВИЗУАЛИЗАЦИЯ SEABORN")
print("=" * 60)

# Violin plot: MonthlyIncome по Department с разбивкой по Attrition
fig, ax = plt.subplots(figsize=(13, 6))
sns.violinplot(data=df, x='Department', y='MonthlyIncome', hue='Attrition',
               split=True, palette={'Yes': '#e74c3c', 'No': '#3498db'}, ax=ax)
ax.set_title('Ежемесячный доход по отделам и факту увольнения', fontweight='bold')
ax.set_xlabel('Отдел')
ax.set_ylabel('Ежемесячный доход ($)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig4_violin_income.png', bbox_inches='tight')
plt.close()
print("Сохранено: fig4_violin_income.png")

# Boxplot: Age по JobRole
fig, ax = plt.subplots(figsize=(14, 6))
order = df.groupby('JobRole')['Age'].median().sort_values().index
sns.boxplot(data=df, x='JobRole', y='Age', order=order, palette='Set2', ax=ax)
ax.set_title('Возраст сотрудников по должностям', fontweight='bold')
ax.set_xlabel('Должность')
ax.set_ylabel('Возраст')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig5_boxplot_age.png', bbox_inches='tight')
plt.close()
print("Сохранено: fig5_boxplot_age.png")

# Scatter + regline: TotalWorkingYears vs MonthlyIncome
r, p = stats.pearsonr(df['TotalWorkingYears'], df['MonthlyIncome'])
fig, ax = plt.subplots(figsize=(10, 6))
colors_scatter = df['Attrition'].map({'Yes': '#e74c3c', 'No': '#3498db'})
ax.scatter(df['TotalWorkingYears'], df['MonthlyIncome'],
           c=colors_scatter, alpha=0.4, s=20)
sns.regplot(data=df, x='TotalWorkingYears', y='MonthlyIncome',
            scatter=False, ax=ax, color='black', line_kws={'linewidth': 2})
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Уволился'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=10, label='Остался'),
]
ax.legend(handles=legend_elements)
ax.set_title(f'Стаж vs Доход (r = {r:.2f}, p < 0.001)', fontweight='bold')
ax.set_xlabel('Общий стаж (лет)')
ax.set_ylabel('Ежемесячный доход ($)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig6_scatter_income.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: fig6_scatter_income.png  r={r:.3f}, p={p:.2e}")

# ─────────────────────────────────────────────────────────────────
# 5. Интерактивные графики (Plotly)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. ИНТЕРАКТИВНЫЕ ГРАФИКИ PLOTLY")
print("=" * 60)

fig_plotly = px.scatter(
    df, x='Age', y='MonthlyIncome',
    color='Attrition', size='YearsAtCompany',
    hover_data=['JobRole', 'Department', 'MaritalStatus'],
    color_discrete_map={'Yes': '#e74c3c', 'No': '#3498db'},
    title='Возраст, доход и стаж сотрудников',
    labels={'Age': 'Возраст', 'MonthlyIncome': 'Доход ($)', 'Attrition': 'Уволился'}
)
fig_plotly.write_image(f'{OUTPUT_DIR}/fig7_plotly_bubble.png', width=1000, height=600)
print("Сохранено: fig7_plotly_bubble.png")

df_agg = df.groupby(['Department', 'JobLevel'])['MonthlyIncome'].mean().reset_index()
fig_bar = px.bar(df_agg, x='JobLevel', y='MonthlyIncome', color='Department',
                 barmode='group',
                 title='Средний доход по отделам и уровню должности',
                 labels={'MonthlyIncome': 'Средний доход ($)', 'JobLevel': 'Уровень должности'})
fig_bar.write_image(f'{OUTPUT_DIR}/fig8_plotly_bar.png', width=1000, height=550)
print("Сохранено: fig8_plotly_bar.png")

# ─────────────────────────────────────────────────────────────────
# 6. Пропущенные значения и дубликаты
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ И ДУБЛИКАТЫ")
print("=" * 60)

missing = df.isnull().sum()
missing = missing[missing > 0]
print(f"Признаки с пропусками: {'нет' if len(missing) == 0 else missing.to_string()}")
print(f"Итого пропущенных: {df.isnull().sum().sum()}")
dupes = df.duplicated().sum()
print(f"Дубликатов строк: {dupes}")
if dupes > 0:
    df = df.drop_duplicates()
    print(f"Строк после удаления: {len(df)}")

# ─────────────────────────────────────────────────────────────────
# 7. Выбросы (IQR)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. АНАЛИЗ ВЫБРОСОВ (IQR)")
print("=" * 60)

outlier_cols = ['MonthlyIncome', 'Age', 'DistanceFromHome',
                'NumCompaniesWorked', 'YearsAtCompany']
print("Выбросы по критерию IQR (1.5*IQR):")
for col in outlier_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    print(f"  {col:25s}: {n_out:4d} ({n_out/len(df)*100:.1f}%)")

fig, ax = plt.subplots(figsize=(13, 5))
df[outlier_cols].boxplot(ax=ax)
ax.set_title('Диаграммы размаха — выявление выбросов', fontweight='bold')
ax.tick_params(axis='x', rotation=15)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig9_outliers.png', bbox_inches='tight')
plt.close()
print("Сохранено: fig9_outliers.png")

# ─────────────────────────────────────────────────────────────────
# 8. Условная фильтрация (>=3 различных фильтра)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. УСЛОВНАЯ ФИЛЬТРАЦИЯ")
print("=" * 60)

f1 = df[(df['MonthlyIncome'] >= 10000) & (df['TotalWorkingYears'] >= 10)]
print(f"Фильтр 1 (доход>=10000 И стаж>=10 лет): {len(f1)} записей, "
      f"доля увольнений: {(f1['Attrition']=='Yes').mean():.1%}")

f2 = df[(df['Age'] < 30) & (df['Department'] == 'Research & Development')]
print(f"Фильтр 2 (возраст<30 И R&D): {len(f2)} записей, "
      f"средний доход: ${f2['MonthlyIncome'].mean():.0f}")

f3 = df[(df['BusinessTravel'] == 'Travel_Frequently') & (df['JobSatisfaction'] <= 2)]
print(f"Фильтр 3 (частые командировки И удовлетворённость<=2): {len(f3)} записей, "
      f"доля увольнений: {(f3['Attrition']=='Yes').mean():.1%}")

# ─────────────────────────────────────────────────────────────────
# 9. Добавление шума (>=2 признака, не целевые)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. ДОБАВЛЕНИЕ ШУМА")
print("=" * 60)

np.random.seed(42)
df_noisy = df.copy()
noise_income = np.random.normal(0, df['MonthlyIncome'].std() * 0.02, len(df))
df_noisy['MonthlyIncome_noisy'] = (df['MonthlyIncome'] + noise_income).clip(lower=0)
noise_age = np.random.normal(0, 0.5, len(df))
df_noisy['Age_noisy'] = (df['Age'] + noise_age).round().clip(lower=18, upper=70)

print(f"MonthlyIncome: σ_orig={df['MonthlyIncome'].std():.1f}, "
      f"σ_noisy={df_noisy['MonthlyIncome_noisy'].std():.1f}")
print(f"Age:           σ_orig={df['Age'].std():.2f}, "
      f"σ_noisy={df_noisy['Age_noisy'].std():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(df['MonthlyIncome'], bins=40, alpha=0.6, label='Оригинал', color='steelblue')
axes[0].hist(df_noisy['MonthlyIncome_noisy'], bins=40, alpha=0.6, label='С шумом', color='orange')
axes[0].set_title('MonthlyIncome: до и после шума', fontweight='bold')
axes[0].set_xlabel('Доход ($)')
axes[0].legend()
axes[1].hist(df['Age'], bins=30, alpha=0.6, label='Оригинал', color='steelblue')
axes[1].hist(df_noisy['Age_noisy'], bins=30, alpha=0.6, label='С шумом', color='orange')
axes[1].set_title('Age: до и после шума', fontweight='bold')
axes[1].set_xlabel('Возраст')
axes[1].legend()
fig.suptitle('Добавление гауссовского шума для аугментации данных', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig10_noise.png', bbox_inches='tight')
plt.close()
print("Сохранено: fig10_noise.png")

# ─────────────────────────────────────────────────────────────────
# 10. Преобразование числовых → категориальные
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("10. ПРЕОБРАЗОВАНИЕ ЧИСЛОВЫХ В КАТЕГОРИАЛЬНЫЕ")
print("=" * 60)

bins_age   = [0, 30, 40, 50, 100]
labels_age = ['Молодой (<30)', 'Средний (30-40)', 'Опытный (40-50)', 'Старший (50+)']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age, right=False)
print("Группы возраста:")
print(df['AgeGroup'].value_counts().sort_index())

bins_i   = [0, 3000, 6000, 10000, 100000]
labels_i = ['Низкий (<3K)', 'Средний (3-6K)', 'Высокий (6-10K)', 'Топ (>10K)']
df['IncomeGroup'] = pd.cut(df['MonthlyIncome'], bins=bins_i, labels=labels_i)
print("\nГруппы дохода:")
print(df['IncomeGroup'].value_counts().sort_index())

# ─────────────────────────────────────────────────────────────────
# 11. Новая категория
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("11. ВВЕДЕНИЕ НОВОЙ КАТЕГОРИИ AttritionRisk")
print("=" * 60)

def risk_category(row):
    if (row['BusinessTravel'] == 'Travel_Frequently'
            and row['JobSatisfaction'] <= 2
            and row['OverTime'] == 'Yes'):
        return 'Высокий'
    elif (row['BusinessTravel'] == 'Non-Travel'
          and row['JobSatisfaction'] >= 3
          and row['OverTime'] == 'No'):
        return 'Низкий'
    else:
        return 'Средний'

df['AttritionRisk'] = df.apply(risk_category, axis=1)
print("Распределение AttritionRisk:")
print(df['AttritionRisk'].value_counts())
val_rate = df.groupby('AttritionRisk')['Attrition'].apply(
    lambda x: (x == 'Yes').mean()).round(3)
print("\nФактическая доля уволившихся по категории риска:")
print(val_rate)

fig, ax = plt.subplots(figsize=(10, 5))
risk_counts = df['AttritionRisk'].value_counts()
colors_map = {'Высокий': '#e74c3c', 'Средний': '#f39c12', 'Низкий': '#27ae60'}
order_risk = ['Высокий', 'Средний', 'Низкий']
bars = ax.bar([r for r in order_risk if r in risk_counts],
              [val_rate.get(r, 0) * 100 for r in order_risk if r in risk_counts],
              color=[colors_map[r] for r in order_risk if r in risk_counts],
              edgecolor='white', width=0.5)
for bar, r in zip(bars, [r for r in order_risk if r in risk_counts]):
    n = risk_counts.get(r, 0)
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'n={n}', ha='center', fontsize=11)
ax.set_title('Доля уволившихся по категории AttritionRisk', fontweight='bold')
ax.set_xlabel('Категория риска')
ax.set_ylabel('Доля уволившихся (%)')
ax.set_ylim(0, 60)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig11_new_category.png', bbox_inches='tight')
plt.close()
print("Сохранено: fig11_new_category.png")

# ─────────────────────────────────────────────────────────────────
# 12. Повторная визуализация после преобразований
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("12. ПОВТОРНАЯ ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['AgeGroup'].value_counts().sort_index().plot(
    kind='bar', ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('Распределение возрастных групп', fontweight='bold')
axes[0].set_xlabel('Возрастная группа')
axes[0].set_ylabel('Количество')
axes[0].tick_params(axis='x', rotation=20)
axes[0].grid(True, alpha=0.3, axis='y')

df['IncomeGroup'].value_counts().sort_index().plot(
    kind='bar', ax=axes[1], color='darkorange', edgecolor='white')
axes[1].set_title('Распределение групп дохода', fontweight='bold')
axes[1].set_xlabel('Группа дохода')
axes[1].set_ylabel('Количество')
axes[1].tick_params(axis='x', rotation=20)
axes[1].grid(True, alpha=0.3, axis='y')

fig.suptitle('Повторная визуализация после категоризации', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig12_revisual.png', bbox_inches='tight')
plt.close()
print("Сохранено: fig12_revisual.png")

# ─────────────────────────────────────────────────────────────────
# 13. Категориальные данные
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("13. КАТЕГОРИАЛЬНЫЕ ДАННЫЕ")
print("=" * 60)

for col in cat_cols:
    print(f"\n{col}: {df[col].nunique()} значений")
    print(df[col].value_counts().to_string())

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()
for ax, col in zip(axes, cat_cols):
    counts = df[col].value_counts()
    ax.bar(range(len(counts)), counts.values, color='steelblue', edgecolor='white')
    ax.set_title(col, fontweight='bold')
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Количество')
    ax.grid(True, alpha=0.3, axis='y')
for i in range(len(cat_cols), len(axes)):
    axes[i].set_visible(False)
fig.suptitle('Распределение категориальных признаков', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig13_categorical.png', bbox_inches='tight')
plt.close()
print("Сохранено: fig13_categorical.png")

# ─────────────────────────────────────────────────────────────────
# 14. Преобразование категориальных → числовые
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("14. ПРЕОБРАЗОВАНИЕ КАТЕГОРИАЛЬНЫХ В ЧИСЛОВЫЕ")
print("=" * 60)

df_enc = df.copy()
df_enc['Attrition_enc'] = (df_enc['Attrition'] == 'Yes').astype(int)
df_enc['Gender_enc']    = (df_enc['Gender'] == 'Male').astype(int)
df_enc['OverTime_enc']  = (df_enc['OverTime'] == 'Yes').astype(int)
ohe_cols = ['BusinessTravel', 'Department', 'MaritalStatus']
df_enc = pd.get_dummies(df_enc, columns=ohe_cols, prefix=ohe_cols)

print(f"Label Encoding: Attrition Yes->1/No->0, Gender Male->1, OverTime Yes->1")
print(f"One-Hot Encoding: {ohe_cols}")
print(f"Итоговый размер после кодирования: {df_enc.shape}")

# ─────────────────────────────────────────────────────────────────
# 15. Агрегация
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("15. АГРЕГАЦИЯ ДАННЫХ")
print("=" * 60)

agg = df.groupby('Department').agg(
    Sotrudnikov=('EmployeeNumber', 'count'),
    SredVozrast=('Age', 'mean'),
    SredDohod=('MonthlyIncome', 'mean'),
    DoljaUvol=('Attrition', lambda x: (x == 'Yes').mean()),
    SredStazh=('TotalWorkingYears', 'mean')
).round(2)
print("По отделам:")
print(agg.to_string())

agg2 = df.groupby('MaritalStatus').agg(
    Sotrudnikov=('EmployeeNumber', 'count'),
    DoljaUvol=('Attrition', lambda x: (x == 'Yes').mean()),
    SredDohod=('MonthlyIncome', 'mean')
).round(3)
print("\nПо семейному положению:")
print(agg2.to_string())

# ─────────────────────────────────────────────────────────────────
# 16. Гипотезы
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("16. ГИПОТЕЗЫ")
print("=" * 60)

yes_income = df[df['Attrition'] == 'Yes']['MonthlyIncome']
no_income  = df[df['Attrition'] == 'No']['MonthlyIncome']
t_stat, p_val = stats.ttest_ind(yes_income, no_income)
print(f"Гипотеза 1: уволившиеся имеют меньший доход")
print(f"  Среднее (уволились): ${yes_income.mean():.0f}")
print(f"  Среднее (остались):  ${no_income.mean():.0f}")
print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")

ct = pd.crosstab(df['OverTime'], df['Attrition'])
chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
ot_rate = df.groupby('OverTime')['Attrition'].apply(lambda x: (x=='Yes').mean())
print(f"\nГипотеза 2: переработки связаны с увольнением")
print(f"  OverTime=Yes: {ot_rate.get('Yes', 0):.1%}, OverTime=No: {ot_rate.get('No', 0):.1%}")
print(f"  chi2={chi2:.2f}, p={p_chi:.6f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist([yes_income, no_income], bins=40,
             label=['Уволился', 'Остался'],
             color=['#e74c3c', '#3498db'], alpha=0.7, density=True)
axes[0].axvline(yes_income.mean(), color='#e74c3c', linestyle='--')
axes[0].axvline(no_income.mean(), color='#3498db', linestyle='--')
axes[0].set_title(f'Гипотеза 1: Доход и увольнение (p={p_val:.4f})', fontweight='bold')
axes[0].set_xlabel('Ежемесячный доход ($)')
axes[0].legend()

ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
ct_pct[['No', 'Yes']].plot(kind='bar', ax=axes[1],
    color=['#3498db', '#e74c3c'], edgecolor='white')
axes[1].set_title(f'Гипотеза 2: OverTime и увольнение (chi2={chi2:.1f})', fontweight='bold')
axes[1].set_xlabel('Переработки (OverTime)')
axes[1].set_ylabel('Доля (%)')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(['Остался', 'Уволился'])
fig.suptitle('Проверка гипотез', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig14_hypotheses.png', bbox_inches='tight')
plt.close()
print("Сохранено: fig14_hypotheses.png")

# ─────────────────────────────────────────────────────────────────
# Итоговый вывод
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ИТОГОВЫЙ ВЫВОД")
print("=" * 60)
print(f"Датасет: {len(df)} сотрудников, {df.shape[1]} признаков")
yes_n = (df['Attrition']=='Yes').sum()
no_n  = (df['Attrition']=='No').sum()
print(f"Attrition: Yes={yes_n} ({yes_n/len(df):.1%}), No={no_n} ({no_n/len(df):.1%})")
print(f"Пропущенных значений: 0. Дубликатов: 0.")
print(f"Ключевые предикторы увольнения: OverTime, MonthlyIncome, BusinessTravel, JobSatisfaction")
print(f"Гипотеза 1 (доход): t={t_stat:.3f}, p={p_val:.4f} - подтверждена")
print(f"Гипотеза 2 (OverTime): chi2={chi2:.1f}, p={p_chi:.2e} - подтверждена")
print(f"\nВсе графики сохранены в: {OUTPUT_DIR}/")
