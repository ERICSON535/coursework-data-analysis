"""
Глава 4. Первичный анализ набора текстовых данных
Датасет: BBC News Classification
Источник: https://www.kaggle.com/datasets/alfathterry91/bbc-news-classification

Инструкция по загрузке:
  1. Скачайте: kaggle datasets download -d alfathterry91/bbc-news-classification
  2. Распакуйте в папку data/
  3. Файл должен называться: data/bbc_news.csv
     (столбцы: category, text)
"""

import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

OUTPUT_DIR = 'plots/chapter4'
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_PATH = 'data/bbc_news.csv'

STOP_WORDS_RU = set()
try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOP_WORDS = set(ENGLISH_STOP_WORDS)
except Exception:
    STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                  'to', 'for', 'of', 'is', 'it', 'was', 'are', 'be', 'has',
                  'had', 'with', 'as', 'by', 'from', 'that', 'this', 'said'}

CATEGORIES = ['sport', 'business', 'politics', 'tech', 'entertainment']
COLORS = {'sport': '#4CAF50', 'business': '#2196F3', 'politics': '#FF5722',
          'tech': '#9C27B0', 'entertainment': '#FF9800'}

# ── 1. Загрузка данных ─────────────────────────────────────────────────────────
print("=" * 60)
print("1. ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
# Попытка найти нужные столбцы
text_col = next((c for c in df.columns if 'text' in c.lower()), df.columns[-1])
cat_col  = next((c for c in df.columns if 'cat' in c.lower() or 'label' in c.lower()
                 or 'class' in c.lower()), df.columns[0])
df = df.rename(columns={text_col: 'text', cat_col: 'category'})
df = df[['category', 'text']].dropna()
df['category'] = df['category'].str.lower().str.strip()

print(f"Размер датасета: {df.shape[0]} статей × {df.shape[1]} столбцов")
print(f"Категории: {sorted(df['category'].unique())}")
print(f"\nПример статьи (первые 300 символов):")
print(df.iloc[0]['text'][:300])

# ── 2. Распределение категорий ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. РАСПРЕДЕЛЕНИЕ КАТЕГОРИЙ")
print("=" * 60)

cat_counts = df['category'].value_counts()
print(cat_counts)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
bars = ax1.bar(cat_counts.index, cat_counts.values,
               color=[COLORS.get(c, 'steelblue') for c in cat_counts.index], edgecolor='white')
ax1.set_title('Количество статей по категориям', fontweight='bold')
ax1.set_xlabel('Категория')
ax1.set_ylabel('Количество статей')
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)

ax2.pie(cat_counts.values, labels=cat_counts.index,
        colors=[COLORS.get(c, 'grey') for c in cat_counts.index],
        autopct='%1.1f%%', startangle=140, pctdistance=0.8)
ax2.set_title('Доля категорий', fontweight='bold')
fig.suptitle('BBC News Classification — распределение категорий',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_category_distribution.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/fig1_category_distribution.png")

# ── 3. Анализ длины текстов ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. ДЛИНА ТЕКСТОВ")
print("=" * 60)

df['char_len']  = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['sent_count'] = df['text'].str.count(r'[.!?]+')

print(df.groupby('category')[['char_len', 'word_count', 'sent_count']].agg(['mean', 'median']).round(1))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for cat in df['category'].unique():
    data = df[df['category'] == cat]
    axes[0].hist(data['word_count'], bins=40, alpha=0.55, label=cat,
                 color=COLORS.get(cat, 'grey'), density=True)
    axes[1].hist(data['char_len'],  bins=40, alpha=0.55, label=cat,
                 color=COLORS.get(cat, 'grey'), density=True)

axes[0].set_title('Распределение длин текстов (слова)', fontweight='bold')
axes[0].set_xlabel('Количество слов')
axes[0].legend(fontsize=8)
axes[1].set_title('Распределение длин текстов (символы)', fontweight='bold')
axes[1].set_xlabel('Количество символов')
axes[1].legend(fontsize=8)
fig.suptitle('Длины статей BBC News по категориям', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_text_lengths.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/fig2_text_lengths.png")

# ── 4. Частотный анализ слов ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. ЧАСТОТНЫЙ АНАЛИЗ СЛОВ")
print("=" * 60)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

df['tokens'] = df['text'].apply(clean_text)
df['vocab_size'] = df['tokens'].apply(lambda x: len(set(x)))

total_vocab = len(set(w for tokens in df['tokens'] for w in tokens))
print(f"Общий объём словаря: {total_vocab} уникальных слов")
print(f"Среднее слов в статье: {df['word_count'].mean():.0f}")

# Топ-20 слов на весь корпус
all_words = [w for tokens in df['tokens'] for w in tokens]
top20 = Counter(all_words).most_common(20)
words, freqs = zip(*top20)

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(list(words), list(freqs), color='steelblue', edgecolor='white')
ax.set_title('Топ-20 наиболее частых слов (стоп-слова исключены)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Частота встречаемости')
ax.invert_yaxis()
for bar, freq in zip(bars, freqs):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            str(freq), va='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_word_frequency.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/fig3_word_frequency.png")

# ── 5. TF-IDF матрица и визуализация топ-термов по категориям ────────────────
print("\n" + "=" * 60)
print("5. TF-IDF МАТРИЦА И ХАРАКТЕРНЫЕ ТЕРМИНЫ ПО КАТЕГОРИЯМ")
print("=" * 60)

vectorizer = TfidfVectorizer(max_features=500, min_df=3, ngram_range=(1, 1),
                              stop_words='english', sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform(df['text'])
print(f"TF-IDF матрица: {tfidf_matrix.shape[0]} документов × {tfidf_matrix.shape[1]} термов")
print(f"Плотность матрицы: {tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.4f}")

feature_names = vectorizer.get_feature_names_out()
print(f"\nПервые 20 термов TF-IDF: {list(feature_names[:20])}")

# Топ-8 TF-IDF термов по категориям и визуализация
print("\nТоп-8 TF-IDF термов по категориям:")
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
tfidf_df['category'] = df['category'].values

cats_sorted = sorted(df['category'].unique())
fig, axes = plt.subplots(1, len(cats_sorted), figsize=(18, 6))
for ax, cat in zip(axes, cats_sorted):
    cat_means = tfidf_df[tfidf_df['category'] == cat][feature_names].mean()
    top8 = cat_means.nlargest(8)
    print(f"  {cat}: {list(top8.index)}")
    ax.barh(list(top8.index), list(top8.values),
            color=COLORS.get(cat, 'steelblue'), edgecolor='white')
    ax.set_title(cat.upper(), fontweight='bold', fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Средний TF-IDF', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
fig.suptitle('Топ-8 TF-IDF термов по категориям BBC News',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig4_tfidf.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/fig4_tfidf.png")

# ── 7. Bag-of-Words матрица ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. BAG-OF-WORDS МАТРИЦА")
print("=" * 60)

cv = CountVectorizer(max_features=3000, min_df=2, stop_words='english')
bow_matrix = cv.fit_transform(df['text'])
print(f"BoW матрица: {bow_matrix.shape[0]} × {bow_matrix.shape[1]}")
print(f"Среднее кол-во ненулевых слов в документе: {bow_matrix.nnz / bow_matrix.shape[0]:.1f}")
print(f"Разреженность: {(1 - bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1])) * 100:.1f}%")

# ── 8. Анализ n-грамм ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. БИГРАММЫ")
print("=" * 60)

bigram_cv = CountVectorizer(ngram_range=(2, 2), max_features=30, stop_words='english')
bigram_matrix = bigram_cv.fit_transform(df['text'])
bigram_sums = np.asarray(bigram_matrix.sum(axis=0)).flatten()
bigrams_sorted = sorted(zip(bigram_cv.get_feature_names_out(), bigram_sums),
                        key=lambda x: -x[1])
print("Топ-15 биграмм:")
for bg, cnt in bigrams_sorted[:15]:
    print(f"  {bg}: {cnt}")

fig, ax = plt.subplots(figsize=(11, 6))
bigs = [bg for bg, _ in bigrams_sorted[:15]]
cnts = [cnt for _, cnt in bigrams_sorted[:15]]
ax.barh(bigs, cnts, color='mediumpurple', edgecolor='white')
ax.set_title('Топ-15 биграмм корпуса BBC News', fontsize=13, fontweight='bold')
ax.set_xlabel('Частота')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig5_bigrams.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/fig5_bigrams.png")

# ── 9. Вывод ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ИТОГОВЫЙ ВЫВОД")
print("=" * 60)
print(f"Датасет содержит {len(df)} статей, 5 категорий, словарь {total_vocab} слов.")
print(f"Средняя длина статьи: {df['word_count'].mean():.0f} слов.")
print(f"TF-IDF матрица: {tfidf_matrix.shape}, плотность очень низкая (разреженность высокая).")
print("Дисбаланс классов минимален (~20% на категорию).")
print("Ключевые слова хорошо разграничивают категории (sport ↔ politics ↔ tech).")
print("Датасет пригоден для задачи многоклассовой классификации текстов.")
print(f"\nВсе графики сохранены в директории: {OUTPUT_DIR}/")
