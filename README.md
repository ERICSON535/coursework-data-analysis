# Курсовой проект: Поиск и первичный анализ наборов данных

**Автор:** Уряшев Кирилл Александрович  
**Группа:** ЕТ-103  
**Направление:** 09.03.03 «Прикладная информатика»  
**Университет:** ЮУрГУ (НИУ), Центр «ВиртУм»  
**Руководитель:** Паршукова Н.Б.  
**Год:** 2026

---

## Описание

Репозиторий содержит Python-скрипты первичного анализа пяти наборов данных разных типов, используемых в задачах искусственного интеллекта.

## Датасеты

| Глава | Тип данных | Датасет | Источник |
|-------|-----------|---------|---------|
| 1 | Табличные | World Happiness Report 2021 | [Kaggle](https://www.kaggle.com/datasets/mathurinache/world-happiness-report-2021) |
| 2 | Временные ряды | Human Activity Recognition (HAR) | [Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones) |
| 3 | Изображения | Chest X-Ray Pneumonia | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| 4 | Текст | BBC News Classification | [Kaggle](https://www.kaggle.com/datasets/alfathterry91/bbc-news-classification) |
| 5 | Аудио | GTZAN Music Genre Classification | [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) |

## Структура репозитория

```
coursework-data-analysis/
├── chapter1_tabular_happiness.py   # Анализ табличных данных
├── chapter2_timeseries_har.py      # Анализ временных рядов
├── chapter3_images_xray.py         # Анализ изображений
├── chapter4_text_bbc.py            # Анализ текстовых данных
├── chapter5_audio_gtzan.py         # Анализ аудиоданных
├── data/                           # Директория для датасетов (скачать с Kaggle)
├── plots/                          # Генерируемые графики
│   ├── chapter1/
│   ├── chapter2/
│   ├── chapter3/
│   ├── chapter4/
│   └── chapter5/
├── requirements.txt
└── README.md
```

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Загрузка датасетов

```bash
# Установите Kaggle API
pip install kaggle

# Настройте токен: положите kaggle.json в ~/.kaggle/

# Скачайте датасеты
kaggle datasets download -d mathurinache/world-happiness-report-2021 -p data/ --unzip
kaggle datasets download -d uciml/human-activity-recognition-with-smartphones -p data/har/ --unzip
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/ --unzip
kaggle datasets download -d alfathterry91/bbc-news-classification -p data/ --unzip
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification -p data/gtzan/ --unzip
```

## Запуск анализа

```bash
python chapter1_tabular_happiness.py
python chapter2_timeseries_har.py
python chapter3_images_xray.py
python chapter4_text_bbc.py
python chapter5_audio_gtzan.py
```

## Требования к окружению

- Python 3.9+
- Зависимости: `requirements.txt`
