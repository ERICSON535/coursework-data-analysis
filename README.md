# Курсовой проект: Поиск и первичный анализ наборов данных

**Автор:** Уряшев Кирилл Александрович  
**Группа:** ЕТ-103  
**Направление:** 09.03.03 «Прикладная информатика»  
**Университет:** ЮУрГУ (НИУ), Центр «ВиртУм»  
**Руководитель:** Паршукова Н.Б.  
**Год:** 2026

---

## Структура проекта

| Файл | Описание | Набор данных |
|------|----------|-------------|
| `chapter1_tabular_happiness.py` | Глава 1. Первичный анализ табличных данных | World Happiness Report 2021 ([Kaggle](https://www.kaggle.com/datasets/mathurinache/world-happiness-report-2021)) |
| `chapter2_timeseries_power.py` | Глава 2. Первичный анализ временных рядов | Power Consumption of Tetouan City ([Kaggle](https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption)) |
| `chapter3_images_xray.py` | Глава 3. Первичный анализ данных с изображениями | Chest X-Ray Images (Pneumonia) ([Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)) |
| `chapter4_text_bbc.py` | Глава 4. Первичный анализ текстовых данных | BBC News Classification ([Kaggle](https://www.kaggle.com/datasets/alfathterry91/bbc-news-classification)) |

---

## Что делают скрипты

Каждый скрипт выполняет **полный цикл первичного анализа данных** для своего типа:

- Загрузка и первичное знакомство с данными
- Визуализация (гистограммы, диаграммы рассеяния, тепловые карты)
- Статистический анализ (описательная статистика, распределения)
- Анализ пропусков и выбросов
- Корреляционный анализ
- Специфические процедуры для каждого типа данных

---

## Зависимости

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install statsmodels                   # Глава 2
pip install Pillow                        # Глава 3
```

---

## Загрузка датасетов

```bash
pip install kaggle
# Положите kaggle.json в ~/.kaggle/

kaggle datasets download -d mathurinache/world-happiness-report-2021 -p data/ --unzip
kaggle datasets download -d fedesoriano/electric-power-consumption -p data/power/ --unzip
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/ --unzip
kaggle datasets download -d alfathterry91/bbc-news-classification -p data/ --unzip
```

---

## Запуск

```bash
python chapter1_tabular_happiness.py
python chapter2_timeseries_power.py
python chapter3_images_xray.py
python chapter4_text_bbc.py
```

> **Примечание:** перед запуском скачайте соответствующий набор данных с Kaggle и укажите путь к нему в начале нужного скрипта (переменная `DATA_PATH` / `DATA_DIR`).
