#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Единый скрипт обработки и анализа корпуса видеотранскриптов Bilibili.

Последовательность блоков:
    1. Очистка и токенизация корпуса (corpus.csv -> corpus_clean.csv)
    2. Фильтрация нерелевантного контента (corpus_clean.csv -> corpus_clean_nogames.csv)
    3. Частотный анализ, конкорданс, коллокации
    4. TF-IDF
    5. LDA (тематическое моделирование)

Все аналитические блоки (3 - 5) работают с отфильтрованным корпусом
corpus_clean_nogames.csv.

Запуск:
    python pipeline.py            # полный прогон
    python pipeline.py --step 3   # запуск отдельного блока

Зависимости: pandas, jieba, scikit-learn, scipy, numpy.
"""

import argparse
import csv
import os
import re
import ssl
import urllib.request
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# Пути к файлам. Рядом со скриптом должен лежать исходный corpus.csv.
# ══════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_CORPUS         = os.path.join(BASE_DIR, "corpus.csv")
CORPUS_CLEAN       = os.path.join(BASE_DIR, "corpus_clean.csv")
CORPUS_NOGAMES     = os.path.join(BASE_DIR, "corpus_clean_nogames.csv")

FREQUENCY_CSV      = os.path.join(BASE_DIR, "frequency.csv")
CONCORDANCE_CSV    = os.path.join(BASE_DIR, "concordance.csv")
COLLOCATIONS_CSV   = os.path.join(BASE_DIR, "collocations.csv")

TFIDF_TOP_WORDS    = os.path.join(BASE_DIR, "tfidf_top_words.csv")
TFIDF_OVERALL      = os.path.join(BASE_DIR, "tfidf_overall.csv")
TFIDF_VOCAB        = os.path.join(BASE_DIR, "tfidf_vocab.csv")
TFIDF_SPARSE       = os.path.join(BASE_DIR, "tfidf_sparse.npz")

LDA_TOPICS         = os.path.join(BASE_DIR, "lda_topics.csv")
LDA_DOCUMENTS      = os.path.join(BASE_DIR, "lda_documents.csv")


# ══════════════════════════════════════════════════════════════════════
# БЛОК 1. Очистка и токенизация
# ══════════════════════════════════════════════════════════════════════
MANUAL_STOPWORDS = {
    "我们", "他们", "什么", "因为", "没有", "现在", "一个", "可能",
    "不要", "觉得", "其实", "时候", "直接", "这样", "怎么", "一定",
    "那个", "还有", "已经", "不能", "需要", "这种", "很多", "里面",
    "这是", "两个", "喜欢", "看到", "一起", "是不是", "不会", "一天",
    "特别", "发现", "之后", "视频", "不用", "一直", "这么", "好像",
    "起来", "最后", "地方", "之前", "房间", "哈哈哈", "爸爸", "一次",
    "一点", "晚上", "开始", "有人", "老师", "我們", "他們", "這個",
    "什麼", "因為", "可以", "沒有", "知道", "現在", "今天", "真的",
    "非常", "覺得", "所以", "但是", "如果", "已經", "還是", "就是",
    "也是", "都是", "只是", "還有", "然后", "那個", "這樣", "時候",
    "東西", "事情", "感覺", "時間", "問題", "兩個", "看看", "孩子",
    "這種", "這裡", "這麼", "好像",
}


def load_stopwords():
    """Авторский стоп-лист + стандартный китайский список goto456."""
    stopwords = set(MANUAL_STOPWORDS)
    url = "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(url, context=ctx) as r:
            for line in r.read().decode("utf-8").splitlines():
                stopwords.add(line.strip())
        print(f"  Стоп-слова загружены: {len(stopwords)}")
    except Exception as e:
        print(f"  Загрузка стоп-слов не удалась: {e}. Используется только ручной список.")
    return stopwords


def clean_text(text, stopwords, jieba):
    """Удаляет всё кроме китайских иероглифов, режет через jieba, убирает стоп-слова."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"[^\u4e00-\u9fff]", " ", text)
    if not text.strip():
        return ""
    words = jieba.cut(text, cut_all=False)
    clean = [w for w in words if w.strip() and w not in stopwords and len(w) > 1]
    return " ".join(clean)


def step_clean_corpus():
    """corpus.csv -> corpus_clean.csv. Токенизация через jieba, удаление стоп-слов."""
    print("\n[БЛОК 1] Очистка и токенизация корпуса")

    import jieba
    jieba.setLogLevel("WARN")

    if not os.path.exists(RAW_CORPUS):
        print(f"  Файл {RAW_CORPUS} не найден. Пропуск блока.")
        return

    df = pd.read_csv(RAW_CORPUS, encoding="utf-8-sig")
    print(f"  Загружено: {len(df)} строк")

    df["transcript"] = (
        df["transcript"].fillna("").astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.replace("\\n", " ", regex=False)
        .str.strip()
    )

    before = len(df)
    df = df[df["transcript"] != ""].copy()
    print(f"  Удалено без транскрипта: {before - len(df)} (осталось {len(df)})")

    stopwords = load_stopwords()
    df["transcript_clean"] = df["transcript"].apply(
        lambda t: clean_text(t, stopwords, jieba)
    )

    before = len(df)
    df = df[df["transcript_clean"].str.strip() != ""].copy()
    print(f"  Удалено с пустым текстом после очистки: {before - len(df)} (осталось {len(df)})")

    df.to_csv(CORPUS_CLEAN, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(f"  Сохранено: {CORPUS_CLEAN}")


# ══════════════════════════════════════════════════════════════════════
# БЛОК 2. Фильтрация игрового и технического контента
# ══════════════════════════════════════════════════════════════════════
# Ключевые слова, указывающие на нерелевантный для исследования контент.
# Первый список - игровая лексика (躺平 в значении игровой стратегии),
# второй - технический контент (сборка ПК, охлаждение), попавший в корпус
# из-за полисемии ключевых запросов.
GAME_KEYWORDS = [
    "游戏", "猎梦者", "发育", "攻略", "模拟", "联机",
    "主播", "版本", "格子", "模式", "玩法", "道具",
]

TECH_KEYWORDS = [
    "水冷", "显卡", "机箱", "分体", "主机", "散热",
]


def step_filter_corpus():
    """corpus_clean.csv -> corpus_clean_nogames.csv. Фильтр по ключевым словам в title."""
    print("\n[БЛОК 2] Фильтрация нерелевантного контента")

    if not os.path.exists(CORPUS_CLEAN):
        print(f"  Файл {CORPUS_CLEAN} не найден. Пропуск блока.")
        return

    df = pd.read_csv(CORPUS_CLEAN, encoding="utf-8-sig")
    print(f"  Загружено: {len(df)} видео")

    pattern = "|".join(GAME_KEYWORDS + TECH_KEYWORDS)
    mask = df["title"].str.contains(pattern, na=False)

    df_excluded = df[mask].reset_index(drop=True)
    df_kept     = df[~mask].reset_index(drop=True)

    print(f"  Исключено видео: {len(df_excluded)}")
    print(f"  Оставлено: {len(df_kept)}")

    df_kept.to_csv(CORPUS_NOGAMES, index=False, encoding="utf-8-sig")
    print(f"  Сохранено: {CORPUS_NOGAMES}")


# ══════════════════════════════════════════════════════════════════════
# БЛОК 3. Частотный анализ, конкорданс, коллокации
# ══════════════════════════════════════════════════════════════════════
SEED_WORDS = [
    "躺平", "内卷", "摆烂", "佛系", "卷", "摸鱼",
    "打工", "工作", "公司", "996", "加班", "工位",
    "压力", "焦虑", "累", "疲惫", "崩溃",
    "年轻人", "社会", "阶层", "奋斗", "努力",
    "放弃", "选择", "自由", "意义", "没意思",
]

CONTEXT_WINDOW = 5   # для конкорданса и коллокаций: слов слева и справа
TOP_N          = 100 # сколько слов сохранять в частотном списке


def step_frequency_analysis():
    """Частоты, конкорданс, коллокации по отфильтрованному корпусу."""
    print("\n[БЛОК 3] Частотный анализ, конкорданс, коллокации")

    if not os.path.exists(CORPUS_NOGAMES):
        print(f"  Файл {CORPUS_NOGAMES} не найден. Пропуск блока.")
        return

    df = pd.read_csv(CORPUS_NOGAMES, encoding="utf-8-sig")
    print(f"  Корпус: {len(df)} видео")

    all_words = []
    for text in df["transcript_clean"].dropna():
        all_words.extend(text.split())
    total = len(all_words)
    print(f"  Токенов: {total}, уникальных: {len(set(all_words))}")

    # ------- Частотный список -------
    freq = Counter(all_words)
    freq_df = pd.DataFrame(freq.most_common(TOP_N), columns=["word", "frequency"])
    freq_df["rank"] = range(1, len(freq_df) + 1)
    # относительная частота в процентах, а не промилле
    freq_df["relative_freq_pct"] = (freq_df["frequency"] / total * 100).round(3)
    freq_df = freq_df[["rank", "word", "frequency", "relative_freq_pct"]]
    freq_df.to_csv(FREQUENCY_CSV, index=False, encoding="utf-8-sig")
    print(f"  Сохранено: {FREQUENCY_CSV}")

    # ------- Конкорданс -------
    rows = []
    for _, row in df.iterrows():
        words = str(row.get("transcript_clean", "")).split()
        for i, w in enumerate(words):
            if w in SEED_WORDS:
                rows.append({
                    "bvid":      row.get("bvid", ""),
                    "title":     row.get("title", ""),
                    "date":      row.get("date", ""),
                    "seed_word": w,
                    "left":      " ".join(words[max(0, i - CONTEXT_WINDOW):i]),
                    "node":      w,
                    "right":     " ".join(words[i + 1:i + 1 + CONTEXT_WINDOW]),
                })
    conc_df = pd.DataFrame(rows)
    conc_df.to_csv(CONCORDANCE_CSV, index=False, encoding="utf-8-sig")
    print(f"  Сохранено: {CONCORDANCE_CSV}  ({len(conc_df)} вхождений)")

    # ------- Коллокации -------
    neighbors = defaultdict(list)
    for text in df["transcript_clean"].dropna():
        words = text.split()
        for i, w in enumerate(words):
            if w in SEED_WORDS:
                window = (
                    words[max(0, i - CONTEXT_WINDOW):i]
                    + words[i + 1:i + 1 + CONTEXT_WINDOW]
                )
                neighbors[w].extend(window)

    coll_rows = []
    for seed, ws in neighbors.items():
        for neighbor, count in Counter(ws).most_common(20):
            if neighbor != seed:
                coll_rows.append({
                    "seed_word":      seed,
                    "seed_frequency": freq.get(seed, 0),
                    "collocation":    neighbor,
                    "co_frequency":   count,
                })
    coll_df = pd.DataFrame(coll_rows)
    coll_df.to_csv(COLLOCATIONS_CSV, index=False, encoding="utf-8-sig")
    print(f"  Сохранено: {COLLOCATIONS_CSV}  ({len(coll_df)} пар)")


# ══════════════════════════════════════════════════════════════════════
# БЛОК 4. TF-IDF
# ══════════════════════════════════════════════════════════════════════
TFIDF_MAX_DF         = 0.85    # отбрасываем слова, встречающиеся в >85% документов
TFIDF_MIN_DF         = 3       # требуется минимум в 3 документах
TFIDF_MAX_FEATURES   = 5000
TFIDF_TOP_PER_DOC    = 10
TFIDF_TOP_OVERALL    = 50


def step_tfidf():
    """TF-IDF по отфильтрованному корпусу."""
    print("\n[БЛОК 4] TF-IDF")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import save_npz

    if not os.path.exists(CORPUS_NOGAMES):
        print(f"  Файл {CORPUS_NOGAMES} не найден. Пропуск блока.")
        return

    df = pd.read_csv(CORPUS_NOGAMES, encoding="utf-8-sig")
    df = df[df["transcript_clean"].notna()]
    df = df[df["transcript_clean"].str.strip() != ""].reset_index(drop=True)
    print(f"  Документов: {len(df)}")

    vec = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        max_df=TFIDF_MAX_DF,
        min_df=TFIDF_MIN_DF,
        max_features=TFIDF_MAX_FEATURES,
    )
    matrix = vec.fit_transform(df["transcript_clean"])
    features = vec.get_feature_names_out()
    print(f"  Матрица: {matrix.shape[0]} документов × {matrix.shape[1]} слов")

    # ------- Топ слов по каждому документу -------
    per_doc = []
    for i in range(matrix.shape[0]):
        row = matrix[i].toarray().flatten()
        top_idx = row.argsort()[::-1][:TFIDF_TOP_PER_DOC]
        words  = [features[j] for j in top_idx if row[j] > 0]
        scores = [round(float(row[j]), 4) for j in top_idx if row[j] > 0]
        per_doc.append({
            "bvid":       df.loc[i, "bvid"]   if "bvid"  in df.columns else i,
            "title":      df.loc[i, "title"]  if "title" in df.columns else "",
            "date":       df.loc[i, "date"]   if "date"  in df.columns else "",
            "top_words":  " | ".join(words),
            "top_scores": " | ".join(map(str, scores)),
        })
    pd.DataFrame(per_doc).to_csv(TFIDF_TOP_WORDS, index=False, encoding="utf-8-sig")
    print(f"  Сохранено: {TFIDF_TOP_WORDS}")

    # ------- Топ слов по корпусу в целом -------
    mean_tfidf = np.asarray(matrix.mean(axis=0)).flatten()
    top_overall = mean_tfidf.argsort()[::-1][:TFIDF_TOP_OVERALL]
    overall = pd.DataFrame({
        "rank":       range(1, TFIDF_TOP_OVERALL + 1),
        "word":       [features[j] for j in top_overall],
        "mean_tfidf": [round(float(mean_tfidf[j]), 5) for j in top_overall],
    })
    overall.to_csv(TFIDF_OVERALL, index=False, encoding="utf-8-sig")
    print(f"  Сохранено: {TFIDF_OVERALL}")
    print("  Топ-10 по корпусу:")
    print(overall.head(10).to_string(index=False))

    # ------- Матрица и словарь -------
    save_npz(TFIDF_SPARSE, matrix)
    pd.Series(features).to_csv(
        TFIDF_VOCAB, index=True, header=["word"], encoding="utf-8-sig"
    )
    print(f"  Сохранено: {TFIDF_SPARSE}, {TFIDF_VOCAB}")


# ══════════════════════════════════════════════════════════════════════
# БЛОК 5. LDA
# ══════════════════════════════════════════════════════════════════════
LDA_N_TOPICS      = 6
LDA_N_TOP_WORDS   = 15
LDA_N_ITER        = 50
LDA_RANDOM_STATE  = 42


def step_lda():
    """Тематическое моделирование LDA по отфильтрованному корпусу."""
    print("\n[БЛОК 5] LDA")

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    if not os.path.exists(CORPUS_NOGAMES):
        print(f"  Файл {CORPUS_NOGAMES} не найден. Пропуск блока.")
        return

    df = pd.read_csv(CORPUS_NOGAMES, encoding="utf-8-sig")
    df = df[df["transcript_clean"].notna()]
    df = df[df["transcript_clean"].str.strip() != ""].reset_index(drop=True)
    print(f"  Документов: {len(df)}")

    vec = CountVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        max_df=0.85, min_df=3, max_features=5000,
    )
    dtm = vec.fit_transform(df["transcript_clean"])
    vocab = vec.get_feature_names_out()
    print(f"  Матрица: {dtm.shape}")

    lda = LatentDirichletAllocation(
        n_components=LDA_N_TOPICS,
        max_iter=LDA_N_ITER,
        learning_method="online",
        random_state=LDA_RANDOM_STATE,
        n_jobs=-1,
    )
    lda.fit(dtm)

    # ------- Топ слов по темам -------
    topic_rows = []
    for i, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[::-1][:LDA_N_TOP_WORDS]
        words = [vocab[j] for j in top_idx]
        print(f"\n  Тема {i+1}: {' | '.join(words)}")
        topic_rows.append({"topic": i + 1, "top_words": " | ".join(words)})
    pd.DataFrame(topic_rows).to_csv(LDA_TOPICS, index=False, encoding="utf-8-sig")
    print(f"\n  Сохранено: {LDA_TOPICS}")

    # ------- Распределение тем по документам -------
    doc_topic = lda.transform(dtm)
    df["dominant_topic"]    = doc_topic.argmax(axis=1) + 1
    df["topic_probability"] = doc_topic.max(axis=1).round(3)
    for i in range(LDA_N_TOPICS):
        df[f"topic_{i+1}_weight"] = doc_topic[:, i].round(3)
    df.to_csv(LDA_DOCUMENTS, index=False, encoding="utf-8-sig")
    print(f"  Сохранено: {LDA_DOCUMENTS}")

    print("\n  Распределение по доминирующим темам:")
    counts = df["dominant_topic"].value_counts().sort_index()
    for t, n in counts.items():
        print(f"    Тема {t}: {n} видео ({n/len(df)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# Оркестрация
# ══════════════════════════════════════════════════════════════════════
STEPS = {
    1: step_clean_corpus,
    2: step_filter_corpus,
    3: step_frequency_analysis,
    4: step_tfidf,
    5: step_lda,
}


def main():
    parser = argparse.ArgumentParser(description="Pipeline обработки корпуса")
    parser.add_argument(
        "--step", type=int, choices=list(STEPS.keys()),
        help="Запустить только один блок (по умолчанию все)."
    )
    args = parser.parse_args()

    if args.step:
        STEPS[args.step]()
    else:
        for fn in STEPS.values():
            fn()

    print("\nГотово.")


if __name__ == "__main__":
    main()
