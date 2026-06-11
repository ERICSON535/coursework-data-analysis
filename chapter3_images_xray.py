"""
Глава 3. Первичный анализ набора данных с изображениями
Датасет: Chest X-Ray Images (Pneumonia)
Источник: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Инструкция по загрузке:
  1. Скачайте: kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
  2. Распакуйте в папку data/chest_xray/
  3. Ожидаемая структура:
       data/chest_xray/train/NORMAL/
       data/chest_xray/train/PNEUMONIA/
       data/chest_xray/test/NORMAL/
       data/chest_xray/test/PNEUMONIA/
       data/chest_xray/val/NORMAL/
       data/chest_xray/val/PNEUMONIA/
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

OUTPUT_DIR = 'plots/chapter3'
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = 'data/chest_xray'
SPLITS = ['train', 'test', 'val']
CLASSES = ['NORMAL', 'PNEUMONIA']
TARGET_SIZE = (224, 224)
random.seed(42)
np.random.seed(42)

# ── 1. Загрузка структуры датасета ───────────────────────────────────────────
print("=" * 60)
print("1. СТРУКТУРА ДАТАСЕТА")
print("=" * 60)

all_images = []
for split in SPLITS:
    for cls in CLASSES:
        folder = os.path.join(DATA_DIR, split, cls)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files:
            all_images.append({'split': split, 'class': cls, 'path': os.path.join(folder, f)})

df = pd.DataFrame(all_images)
print(f"Всего изображений: {len(df)}")
print("\nРаспределение по split и классу:")
pivot = df.groupby(['split', 'class']).size().unstack(fill_value=0)
print(pivot)

# ── 2. Распределение классов ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. РАСПРЕДЕЛЕНИЕ КЛАССОВ")
print("=" * 60)

class_counts = df['class'].value_counts()
print(class_counts)
ratio = class_counts[0] / class_counts[1]
print(f"Соотношение классов: {ratio:.2f}:1 ({'дисбаланс классов' if ratio > 1.5 else 'сбалансировано'})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#2196F3', '#F44336']
ax1.bar(class_counts.index, class_counts.values, color=colors, edgecolor='white', width=0.5)
ax1.set_title('Распределение классов (весь датасет)', fontweight='bold')
ax1.set_ylabel('Количество изображений')
for i, (cls, cnt) in enumerate(class_counts.items()):
    ax1.text(i, cnt + 30, str(cnt), ha='center', fontsize=10, fontweight='bold')

split_counts = df.groupby(['split', 'class']).size().reset_index(name='count')
splits_order = ['train', 'val', 'test']
x = np.arange(len(splits_order))
width = 0.35
for i, cls in enumerate(CLASSES):
    vals = [split_counts.query(f"split=='{s}' and `class`=='{cls}'")['count'].sum()
            for s in splits_order]
    ax2.bar(x + i * width, vals, width, label=cls, color=colors[i], edgecolor='white')
ax2.set_xticks(x + width / 2)
ax2.set_xticklabels(splits_order)
ax2.set_title('Распределение по выборкам', fontweight='bold')
ax2.set_ylabel('Количество изображений')
ax2.legend()
# fig.suptitle removed (caption shown below figure in document)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_class_distribution.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/fig1_class_distribution.png")

# ── 3. Примеры изображений ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. ПРИМЕРЫ ИЗОБРАЖЕНИЙ")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for row, cls in enumerate(CLASSES):
    samples = df[(df['class'] == cls) & (df['split'] == 'train')].sample(5, random_state=42)
    for col, (_, row_data) in enumerate(samples.iterrows()):
        img = Image.open(row_data['path']).convert('L')
        axes[row, col].imshow(img, cmap='bone')
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(cls, fontsize=11, fontweight='bold', rotation=90, labelpad=10)
# fig.suptitle removed (caption shown below figure in document)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_sample_images.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/fig2_sample_images.png")

# ── 4. Анализ размеров изображений ───────────────────────────────────────────
print("\n" + "=" * 60)
print("4. АНАЛИЗ РАЗМЕРОВ ИЗОБРАЖЕНИЙ")
print("=" * 60)

sample_paths = df.sample(min(300, len(df)), random_state=42)['path'].tolist()
widths, heights = [], []
for p in sample_paths:
    try:
        with Image.open(p) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    except Exception:
        pass

print(f"Ширина: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
print(f"Высота: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.hist(widths, bins=30, color='steelblue', edgecolor='white')
ax1.set_title('Распределение ширины изображений', fontweight='bold')
ax1.set_xlabel('Пикселей')
ax1.set_ylabel('Частота')
ax1.axvline(np.mean(widths), color='red', linestyle='--', label=f'Среднее: {np.mean(widths):.0f}')
ax1.legend()

ax2.hist(heights, bins=30, color='darkorange', edgecolor='white')
ax2.set_title('Распределение высоты изображений', fontweight='bold')
ax2.set_xlabel('Пикселей')
ax2.set_ylabel('Частота')
ax2.axvline(np.mean(heights), color='red', linestyle='--', label=f'Среднее: {np.mean(heights):.0f}')
ax2.legend()
# fig.suptitle removed (caption shown below figure in document)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_image_sizes.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/fig3_image_sizes.png")

# ── 5. Распределение яркости пикселей ────────────────────────────────────────
print("\n" + "=" * 60)
print("5. РАСПРЕДЕЛЕНИЕ ЯРКОСТИ ПИКСЕЛЕЙ")
print("=" * 60)

pixel_stats = {}
for cls in CLASSES:
    subset = df[(df['class'] == cls) & (df['split'] == 'train')].sample(
        min(50, len(df[df['class'] == cls])), random_state=42)
    all_pixels = []
    for path in subset['path']:
        try:
            img = Image.open(path).convert('L').resize(TARGET_SIZE)
            all_pixels.extend(np.array(img).flatten().tolist())
        except Exception:
            pass
    pixel_stats[cls] = np.array(all_pixels)
    print(f"{cls}: mean={np.mean(all_pixels):.2f}, std={np.std(all_pixels):.2f}, "
          f"min={min(all_pixels)}, max={max(all_pixels)}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (cls, pixels) in zip(axes, pixel_stats.items()):
    ax.hist(pixels, bins=64, color='#2196F3' if cls == 'NORMAL' else '#F44336',
            alpha=0.75, density=True, edgecolor='none')
    ax.axvline(np.mean(pixels), color='black', linestyle='--',
               linewidth=2, label=f'Среднее: {np.mean(pixels):.1f}')
    ax.set_title(f'Гистограмма яркости — {cls}', fontweight='bold')
    ax.set_xlabel('Значение пикселя (0–255)')
    ax.set_ylabel('Плотность')
    ax.legend()
# fig.suptitle removed (caption shown below figure in document)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig4_pixel_histograms.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/fig4_pixel_histograms.png")

# ── 6. Средние изображения по классу ─────────────────────────────────────────
print("\n" + "=" * 60)
print("6. СРЕДНИЕ ИЗОБРАЖЕНИЯ ПО КЛАССУ")
print("=" * 60)

mean_images = {}
for cls in CLASSES:
    subset = df[(df['class'] == cls) & (df['split'] == 'train')].sample(
        min(100, len(df[df['class'] == cls])), random_state=42)
    imgs = []
    for path in subset['path']:
        try:
            img = np.array(Image.open(path).convert('L').resize(TARGET_SIZE), dtype=np.float32)
            imgs.append(img)
        except Exception:
            pass
    if imgs:
        mean_images[cls] = np.mean(imgs, axis=0)
        print(f"{cls}: среднее усредн. изображение по {len(imgs)} снимкам")

if len(mean_images) == 2:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (cls, img) in zip(axes[:2], mean_images.items()):
        im = ax.imshow(img, cmap='bone', vmin=0, vmax=255)
        ax.set_title(f'Среднее изображение\n{cls}', fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    diff = mean_images['PNEUMONIA'] - mean_images['NORMAL']
    im3 = axes[2].imshow(diff, cmap='RdBu_r')
    axes[2].set_title('Разность (PNEUMONIA − NORMAL)', fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # fig.suptitle removed (caption shown below figure in document)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig5_mean_images.png', bbox_inches='tight')
    plt.close()
    print(f"Сохранено: {OUTPUT_DIR}/fig5_mean_images.png")

# ── 7. Контраст и среднее по изображению ─────────────────────────────────────
print("\n" + "=" * 60)
print("7. КОНТРАСТ И СРЕДНЯЯ ЯРКОСТЬ ИЗОБРАЖЕНИЙ")
print("=" * 60)

brightness_data = []
for cls in CLASSES:
    subset = df[(df['class'] == cls) & (df['split'] == 'train')].sample(
        min(200, len(df[df['class'] == cls])), random_state=42)
    for path in subset['path']:
        try:
            img = np.array(Image.open(path).convert('L').resize(TARGET_SIZE), dtype=np.float32)
            brightness_data.append({'class': cls, 'mean': img.mean(), 'std': img.std()})
        except Exception:
            pass

br_df = pd.DataFrame(brightness_data)
print(br_df.groupby('class')[['mean', 'std']].agg(['mean', 'std']).round(2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for cls, color in zip(CLASSES, ['#2196F3', '#F44336']):
    data = br_df[br_df['class'] == cls]
    ax1.hist(data['mean'], bins=30, alpha=0.65, label=cls, color=color, edgecolor='none')
    ax2.hist(data['std'],  bins=30, alpha=0.65, label=cls, color=color, edgecolor='none')
ax1.set_title('Средняя яркость изображений', fontweight='bold')
ax1.set_xlabel('Среднее значение пикселя')
ax1.legend()
ax2.set_title('Стандартное отклонение (контраст)', fontweight='bold')
ax2.set_xlabel('Стандартное отклонение яркости')
ax2.legend()
# fig.suptitle removed (caption shown below figure in document)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig6_brightness_contrast.png', bbox_inches='tight')
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/fig6_brightness_contrast.png")

# ── 8. Вывод ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ИТОГОВЫЙ ВЫВОД")
print("=" * 60)
print(f"Датасет содержит {len(df)} изображений в 2 классах.")
print(f"Соотношение классов: PNEUMONIA:{class_counts.get('PNEUMONIA',0)} / NORMAL:{class_counts.get('NORMAL',0)}")
print("Изображения имеют разные размеры, требуется ресайз (например, 224×224).")
print("Наблюдается дисбаланс классов (больше снимков с пневмонией в обучающей выборке).")
print("Средняя яркость пневмонийных снимков несколько выше — лёгкие более непрозрачны.")
print("Данные пригодны для бинарной классификации методами глубокого обучения (CNN).")
print(f"\nВсе графики сохранены в директории: {OUTPUT_DIR}/")

