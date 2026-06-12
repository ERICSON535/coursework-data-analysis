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
| `chapter1_tabular_happiness.py` | Глава 1. Первичный анализ табличных данных | IBM HR Analytics Employee Attrition ([Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)) |
| `chapter2_timeseries_power.py` | Глава 2. Первичный анализ данных временных рядов | Power Consumption of Tetouan City ([Kaggle](https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption)) |
| `chapter3_images_xray.py` | Глава 3. Первичный анализ данных изображений | Chest X-Ray Images (Pneumonia) ([Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)) |
| `chapter4_text_bbc.py` | Глава 4. Первичный анализ текстовых данных | BBC News Classification ([Kaggle](https://www.kaggle.com/datasets/alfathterry91/bbc-news-classification)) |

---

## Описание датасетов

### Глава 1 — Табличные данные: IBM HR Analytics Employee Attrition
- **Источник:** Kaggle (pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Задача:** бинарная классификация — предсказание факта увольнения сотрудника (Attrition)
- **Объём:** 1 470 сотрудников, 35 признаков
- **Лицензия:** DbCL-1.0

### Глава 2 — Временные ряды: Power Consumption of Tetouan City
- **Источник:** Kaggle (fedesoriano/electric-power-consumption)
- **Задача:** прогнозирование потребления электроэнергии (SARIMA/LSTM)
- **Объём:** 52 417 записей (каждые 10 минут, 2017 год), 3 зоны

### Глава 3 — Изображения: Chest X-Ray (Pneumonia)
- **Источник:** Kaggle (paultimothymooney/chest-xray-pneumonia)
- **Задача:** бинарная классификация — пневмония / норма (CNN)
- **Объём:** 5 571 изображение (JPEG, chest X-ray)

### Глава 4 — Тексты: BBC News Classification
- **Источник:** Kaggle (alfathterry91/bbc-news-classification)
- **Задача:** многоклассовая классификация новостей (5 категорий)
- **Объём:** 2 225 статей, 5 тематических категорий

---

## Что делают скрипты

Каждый скрипт выполняет **полный цикл первичного анализа данных** для своего типа:

- Загрузка и первичное знакомство с данными
- Визуализация (гистограммы, диаграммы рассеяния, тепловые карты)
- Статистический анализ (описательная статистика, распределения)
- Анализ пропущенных значений и выбросов
- Корреляционный анализ
- Специфические процедуры для каждого типа данных:
  - Гл. 1: IQR-выбросы, Seaborn violin/boxplot, Plotly, добавление шума, t-тест, χ²-тест
  - Гл. 2: сезонная декомпозиция (STL), SNR, ACF/PACF, тест Дики–Фуллера (ADF)
  - Гл. 3: распределение классов, яркость/контраст (mean/std пикселей), оценка качества разметки
  - Гл. 4: TF-IDF, Bag-of-Words, биграммы, информационный поиск по корпусу

---

## Зависимости

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly
pip install statsmodels                   # Глава 2 (STL, ADF)
pip install Pillow                        # Глава 3 (обработка изображений)
pip install nltk                          # Глава 4 (стоп-слова, стемминг)
```

---

## Загрузка датасетов

```bash
pip install kaggle
# Положите kaggle.json в ~/.kaggle/

kaggle datasets download -d pavansubhasht/ibm-hr-analytics-attrition-dataset -p data/ --unzip
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

> **Примечание:** перед запуском скачайте соответствующий набор данных с Kaggle и укажите путь к нему в начале скрипта (переменная `DATA_PATH` / `DATA_DIR`).

---

## Результаты

Все графики сохраняются в папку `plots/`:
- `plots/chapter1/` — анализ HR-данных
- `plots/chapter2/` — анализ временных рядов
- `plots/chapter3/` — анализ рентгеновских снимков
- `plots/chapter4/` — анализ новостных текстов
