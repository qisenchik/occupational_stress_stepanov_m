# Tangping Bilibili Corpus

Корпусно-дискурсивное исследование трудового стресса и феномена **Тан Пин** (躺平) на материале видеоблогов китайской платформы Bilibili (2020–2025).

Проектная работа по курсу «Цифровые методы в востоковедении».
НИУ ВШЭ, 2026.

---

## О чём проект

Эмпирическая база — специализированный корпус транскриптов пользовательских видео с Bilibili, охватывающий период с января 2020 по декабрь 2025 года. Финальный рабочий подкорпус: **392 видеозаписи**, **180 304 словоупотребления**, **49 556 уникальных токенов**.

Аналитический пайплайн:

1. **Сбор метаданных** через программный интерфейс Bilibili (с WBI-подписью)
2. **Извлечение текста** — субтитры через API + транскрипция аудио через Whisper (`tiny`)
3. **Очистка и токенизация** — `jieba`, регулярные выражения, объединённый стоп-лист (goto456 + авторский)
4. **Содержательная фильтрация** — лексический фильтр по заголовкам (исключение игрового и технического контента)
5. **Частотный анализ** — `collections.Counter`
6. **Конкорданс KWIC** — окно 5 слов, 27 опорных слов в 4 кластерах
7. **Коллокационный анализ** — топ-20 соседей для каждой опорной единицы
8. **TF-IDF** — `sklearn.feature_extraction.text.TfidfVectorizer`, матрица 392 × 5000
9. **LDA** — `sklearn.decomposition.LatentDirichletAllocation`, 6 тем

Главный аналитический файл — **[`analysis.ipynb`](analysis.ipynb)** — Jupyter-ноутбук с пошаговым исполнением всех блоков на отфильтрованном корпусе и визуализацией результатов.

## Установка

```bash
git clone https://github.com/USERNAME/tangping-bilibili-corpus.git
cd tangping-bilibili-corpus
pip install -r requirements.txt
```

## Запуск

**Вариант 1 — Jupyter-ноутбук (рекомендуется для воспроизведения анализа):**

```bash
jupyter notebook analysis.ipynb
```

**Вариант 2 — единый скрипт `pipeline.py`:**

```bash
python pipeline.py            # полный пайплайн
python pipeline.py --step 3   # только частотный анализ
python pipeline.py --step 4   # только TF-IDF
python pipeline.py --step 5   # только LDA
```

## Структура репозитория

```
tangping-bilibili-corpus/
├── README.md                  # этот файл
├── requirements.txt           # зависимости с версиями
├── .gitignore                 # исключения (cookies, аудио, кэши)
├── analysis.ipynb             # главный аналитический ноутбук с выходами
├── pipeline.py                # единый аналитический скрипт (CLI)
├── data/
│   ├── corpus_clean_nogames.csv   # отфильтрованный корпус (392 видео)
│   ├── frequency.csv              # частотный список (топ-100)
│   ├── concordance.csv            # KWIC по 27 опорным словам
│   ├── collocations.csv           # коллокационные профили
│   ├── tfidf_overall.csv          # топ-50 слов по среднему TF-IDF
│   ├── tfidf_top_words.csv        # топ-10 слов по каждому документу
│   ├── tfidf_sparse.npz           # разреженная матрица TF-IDF
│   ├── tfidf_vocab.csv            # словарь TF-IDF
│   ├── lda_topics.csv             # топ-15 слов для каждой из 6 тем
│   └── lda_documents.csv          # распределение тем по документам
├── stopwords/
│   └── manual_stopwords.txt       # авторский ручной перечень
└── docs/
    └── Proekt_tsifrovye_metody_Stepanov_metodichka.docx   # методичка
```

## Ключевые библиотеки

| Библиотека | Назначение | Документация |
|------------|------------|--------------|
| `requests` | HTTP-запросы к API Bilibili | [requests.readthedocs.io](https://requests.readthedocs.io) |
| `pandas` | Табличные данные, CSV | [pandas.pydata.org](https://pandas.pydata.org) |
| `jieba` | Сегментация китайского текста | [github.com/fxsjy/jieba](https://github.com/fxsjy/jieba) |
| `scikit-learn` | TF-IDF, LDA | [scikit-learn.org](https://scikit-learn.org) |
| `scipy` | Разреженные матрицы | [scipy.org](https://scipy.org) |
| `openai-whisper` | Автоматическая транскрипция аудио | [github.com/openai/whisper](https://github.com/openai/whisper) |
| `yt-dlp` | Скачивание аудиодорожек с Bilibili | [github.com/yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp) |

## Опорные слова исследования

27 единиц в четырёх смысловых кластерах:

- **Ядро феномена:** 躺平, 内卷, 摆烂, 佛系, 卷, 摸鱼
- **Труд и занятость:** 打工, 工作, 公司, 996, 加班, 工位
- **Психологическое состояние:** 压力, 焦虑, 累, 疲惫, 崩溃
- **Социальный контекст:** 年轻人, 社会, 阶层, 奋斗, 努力, 放弃, 选择, 自由, 意义, 没意思

## Воспроизводимость

`random_state=42` зафиксирован в LDA. При тех же входных данных и параметрах результат идентичен. Полное обоснование параметров моделей — в `docs/Proekt_tsifrovye_metody_Stepanov_metodichka.docx`.

## Безопасность

Cookies сессии Bilibili (`SESSDATA`, `bili_jct`, `DedeUserID`) дают полный доступ к аккаунту. Никогда не коммитьте их в репозиторий. См. `.gitignore` — там перечислены файлы, которые не должны попадать в публичный код.

## Лицензия

MIT.

## Автор

Степанов Максим, НИУ ВШЭ, 2026.
