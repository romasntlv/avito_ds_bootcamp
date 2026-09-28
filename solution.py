# %% [markdown]
# # Avito: кандидатогенерация для поиска услуг
#
# Задача: по поисковому запросу отобрать из корпуса объявлений до 50 кандидатов на дальнейшее
# ранжирование. Метрика — Recall@50 (доля релевантных объявлений, попавших в топ-50, усреднённая
# по запросам).
#
# ## Подход
#
# Решение — двухэтапный каскад.
#
# **1. Генерация кандидатов.** Объединяются два независимых источника:
# - BM25 по леммам (заголовок + параметры + описание объявления) — точные и словоформенные совпадения;
# - косинусное сходство эмбеддингов, дообученных на данных задачи, — синонимы и смысловые совпадения
#   («муж на час» → «мастер на час»).
#
# Из каждого источника берётся top-150, объединение даёт пул до ~300 кандидатов на запрос. Отбор в
# top-150 идёт не по сырому скору источника, а с добавлением гео-приора и совпадения «вида услуги»:
# 83% выбранных объявлений находятся в той же локации, что и запрос, а корпус охватывает всю страну —
# без поправки на локацию нужное объявление вытесняется из top-150 текстуально похожими объявлениями
# из других городов и до этапа ранжирования просто не доходит.
#
# **2. Ранжирование.** LightGBM (`lambdarank`, группировка по запросу) переранжирует пул по набору
# признаков (BM25, эмбеддинги, гео, вид услуги, характеристики объявления) и обучается на реальных
# парах «запрос → выбранное объявление» из train.
#
# ## Данные и признаки
#
# Из `search_query` + `search_infm_params_text` строится единый текст запроса, из `item_title_raw` +
# `item_infm_params_text` + `item_description_raw` — текст объявления (поле параметров объявления
# зашумлено служебными полями вроде графика работы и способа оплаты, из него регулярками извлекаются
# только вид/тип услуги, названия конкретных услуг и адрес). Локация, вид услуги, рейтинг, число
# отзывов, цена и флаги объявления идут в признаки ранжирования напрямую.
#
# ## Открытые компоненты
#
# - `sentence-transformers` + модель `deepvk/USER-base` — база для дообучаемых эмбеддингов;
# - `pymorphy3` — морфологический анализатор русского языка (лемматизация для BM25), офлайн;
# - `LightGBM` — градиентный бустинг для ранжирования.
#
# Все компоненты разворачиваются локально, обращений к внешним API на инференсе нет.
#
# ## Валидация
#
# Из train выделяется 2500 запросов для локальной оценки; тексты выбираются равномерно (не по
# частоте), что воспроизводит статистики бенчмарка — доля текстов запроса, встречающихся в остальном
# train (~37%), и среднюю длину запроса (~3.2 слова). Корпус для валидации — 189 212 объявлений
# (выбранные + случайные из train), как и в бенчмарке. Метрика считается в двух точках: потолок пула
# кандидатов (что теряется на генерации кандидатов) и итоговый Recall@50 после LightGBM (что
# дополнительно теряет ранжирование) — это разделяет два разных источника ошибок и подсказывает, что
# оптимизировать дальше.
#
# ## Воспроизводимость
#
# Контрастивное дообучение на GPU не гарантирует битовой идентичности между запусками даже при
# фиксированном сиде. Чтобы результат воспроизводился, по умолчанию ноутбук **загружает уже
# дообученные веса** (`TRAIN_MODEL = False`, путь — `WEIGHTS_PATH`) и детерминированно пересчитывает
# всё остальное. Код дообучения приведён полностью и включается флагом `TRAIN_MODEL = True` — он
# воспроизводит методику, но не гарантирует байт-в-байт те же веса.

# %%
import gc
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

DATA_DIR = Path('data')
OUT_DIR = Path('out')
OUT_DIR.mkdir(exist_ok=True)

TRAIN_MODEL = False              # True — дообучить заново (несколько часов на GPU), False — загрузить веса
WEIGHTS_PATH = Path('model_ft')  # путь к сохранённым дообученным весам

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# %% [markdown]
# ## 1. Тексты запроса и объявления
#
# Описание объявления обрезается до 400 символов сразу при загрузке — оно доминирует по объёму среди
# текстовых полей (в среднем 1405 символов, максимум 8494), и без обрезки на 497k строк train
# многократно раздувает память при каждой копии датафрейма.

# %%
DESC_CHARS = 400
QUERY_PREFIX, ITEM_PREFIX = 'query: ', 'passage: '  # префиксы, с которыми обучена базовая модель

train = pd.read_parquet(DATA_DIR / 'train.parquet')
bench_q = pd.read_parquet(DATA_DIR / 'benchmark_queries.parquet')
bench_items = pd.read_parquet(DATA_DIR / 'benchmark_items.parquet')

for df in (train, bench_items):
    df['item_description_raw'] = df['item_description_raw'].fillna('').str.slice(0, DESC_CHARS)

SEARCH_KEY = ['search_query', 'search_location_id', 'search_is_delivery_search',
              'search_infm_params_text', 'search_category']
for df in (train, bench_q):
    df['search_infm_params_text'] = df['search_infm_params_text'].fillna('')
    df['search_query'] = df['search_query'].fillna('')

# Один поиск = уникальная комбинация признаков запроса; все выбранные в нём объявления — релевантные.
train['gid'] = train.groupby(SEARCH_KEY, sort=False, dropna=False).ngroup()

# Поле параметров объявления содержит полезное (вид/тип услуги, адрес, названия услуг) вперемешку с
# шумом (график работы, способ оплаты, состав бригады). Выделяем полезные поля регулярками по границам
# известных ключей.
_NEXT_KEYS = ['Вид услуги', 'Тип услуги', 'Место оказания услуг', 'Тип стоимости', 'Начальная цена',
              'Работа по договору', 'Работаете', 'Гарантия', 'Время', 'График', 'Как вы работаете',
              'Где вы оказываете', 'Ваши клиенты', 'Кто оказывает', 'Марка', 'Модель', 'Опыт', 'Бригада',
              'Название услуги', 'Услуга', 'Стоимость', 'Продолжительность', 'Онлайн-запись', 'Рейтинг',
              'Проживание', 'Берёте', 'Предоплата', 'Рабочие дни', 'Дни', 'Дополнительно', 'Куда выезжаете',
              'Транспорт', 'Число мест', 'Минимальное', 'Чем вы', 'Тип транспорта', 'Поездка', 'Признак',
              'Выезд', 'Исполнителей', 'Тип помещения']
_LOOKAHEAD = r'(?=\s(?:' + '|'.join(map(re.escape, _NEXT_KEYS)) + r')\b|$)'


def _field(key):
    return re.compile(re.escape(key) + r'\b(.*?)' + _LOOKAHEAD)


RE_VID = _field('Вид услуги')
RE_TIP = re.compile(r'Тип услуги(?! автосервиса)\b(.*?)' + _LOOKAHEAD)
RE_TIP_AUTO = _field('Тип услуги автосервиса')
RE_PLACE = _field('Место оказания услуг')
RE_SERVICE = re.compile(r'(?:Услуга|Название услуги)\b(.*?)' + _LOOKAHEAD)


def _first(rx, s):
    m = rx.search(s)
    return m.group(1).strip() if m else ''


def parse_vid(s):
    """Вид услуги — верхний уровень рубрикатора; пустая строка, если не указан."""
    return _first(RE_VID, s or '')


def compact_item_params(s):
    """Вид/тип услуги, адрес и до 8 названий конкретных услуг из зашумлённого поля параметров."""
    s = re.sub(r'\s+', ' ', s or '')
    parts = []
    for v in (_first(RE_VID, s), _first(RE_TIP, s), _first(RE_TIP_AUTO, s)):
        if v and v not in parts:
            parts.append(v)
    services = [m.group(1).strip() for m in RE_SERVICE.finditer(s)]
    services = [v for v in services if v and v != 'Своя услуга']
    if services:
        parts.append('Услуги: ' + ', '.join(list(dict.fromkeys(services))[:8]))
    place = _first(RE_PLACE, s)
    if place:
        parts.append(place)
    return '. '.join(parts)


def _clean(col):
    return col.fillna('').astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()


def build_query_texts(df):
    q, p = _clean(df['search_query']), _clean(df['search_infm_params_text'])
    return [QUERY_PREFIX + t for t in np.where(p != '', q + '. Фильтры: ' + p, q)]


def build_item_texts(df):
    title = _clean(df['item_title_raw'])
    params = df['item_infm_params_text'].fillna('').map(compact_item_params)
    desc = _clean(df['item_description_raw']).str.slice(0, DESC_CHARS)
    return [ITEM_PREFIX + t for t in title + '. ' + params + '. ' + desc]


for df in (train, bench_items):
    df['item_vid'] = df['item_infm_params_text'].map(parse_vid)
for df in (train, bench_q):
    df['search_vid'] = df['search_infm_params_text'].map(parse_vid)

train['item_text'] = build_item_texts(train)
bench_items['item_text'] = build_item_texts(bench_items)
train['query_text'] = build_query_texts(train)
bench_q['query_text'] = build_query_texts(bench_q)

train = train.drop(columns=['item_title_raw', 'item_description_raw', 'item_infm_params_text',
                            'search_infm_params_text'])
bench_items = bench_items.drop(columns=['item_title_raw', 'item_description_raw', 'item_infm_params_text'])
bench_q = bench_q.drop(columns=['search_infm_params_text'])
gc.collect()

# %% [markdown]
# ## 2. Локальная валидация
#
# 2500 запросов из train, тексты выбираются равномерно (не пропорционально частоте) — так доля
# текстов, встречающихся в остальном train, получается ~37% и средняя длина запроса ~3.1 слова, что
# совпадает со статистиками бенчмарка. Корпус для валидации — 189 212 объявлений (выбранные + случайные
# из train), как и в бенчмарке; выбранные объявления удаляются из обучающей части.

# %%
N_VAL_QUERIES = 2500
CORPUS_SIZE = 189_212


def make_validation(train, n_val_queries, corpus_size, seed):
    groups = train.drop_duplicates('gid')[['gid', 'search_query']]
    one_per_text = groups.sample(frac=1.0, random_state=seed).drop_duplicates('search_query')
    val_groups = one_per_text.sample(n=n_val_queries, random_state=seed)

    is_val = train.gid.isin(set(val_groups.gid))
    val_rows = train[is_val]
    val_pos_items = set(val_rows.item_id)
    train_mask = (~is_val) & (~train.item_id.isin(val_pos_items))

    agg_cols = ['search_query', 'search_location_id', 'search_is_delivery_search', 'search_category',
                'search_vid', 'query_text']
    val_q = (val_rows.groupby('gid', sort=False)
             .agg(**{c: (c, 'first') for c in agg_cols}, positives=('item_id', lambda x: sorted(set(x))))
             .reset_index())
    val_q['seen_text'] = val_q.search_query.isin(set(train.search_query[train_mask]))

    items_all = train.drop_duplicates('item_id')
    pool = items_all[~items_all.item_id.isin(val_pos_items)]
    n_distr = max(0, corpus_size - len(val_pos_items))
    distr = pool.sample(n=min(n_distr, len(pool)), random_state=seed)
    corpus = pd.concat([items_all[items_all.item_id.isin(val_pos_items)], distr], ignore_index=True)
    item_cols = [c for c in train.columns if c.startswith('item_')]
    return val_q, corpus[item_cols].reset_index(drop=True), train_mask.values


val_q, val_corpus, train_mask = make_validation(train, N_VAL_QUERIES, CORPUS_SIZE, SEED)
fit_rows = train[train_mask]
print(f'val queries: {len(val_q)}, val corpus: {len(val_corpus)}, train rows for fit: {train_mask.sum()}')

# %% [markdown]
# ## 3. Гео-приор и совпадение вида услуги
#
# Гео-приор — P(локация объявления | локация поиска) по train, со сглаживанием глобальным
# распределением локаций и псевдо-счётчиком на «ту же локацию» для запросов из непредставленных в
# обучении локаций. Вид услуги: если он указан в фильтрах запроса, в 98% случаев совпадает с видом
# выбранного объявления.

# %%
class GeoPrior:
    def __init__(self, search_locs, item_locs, smoothing=1.0, self_pseudo=3.0):
        self.counts = pd.crosstab(pd.Series(search_locs, name='s'), pd.Series(item_locs, name='i'))
        self.marg = pd.Series(item_locs).value_counts(normalize=True)
        self.smoothing, self.self_pseudo = smoothing, self_pseudo

    def logp_matrix(self, q_locs, i_locs):
        """log P(i_loc | q_loc), матрица [len(q_locs), len(i_locs)] для уникальных локаций."""
        marg = self.marg.reindex(i_locs).fillna(0).values + 1e-6
        marg = marg / marg.sum()
        c = self.counts.reindex(index=q_locs, columns=i_locs).fillna(0).values.astype(np.float64)
        i_pos = {loc: j for j, loc in enumerate(i_locs)}
        for r, loc in enumerate(q_locs):
            if loc in i_pos:
                c[r, i_pos[loc]] += self.self_pseudo
        p = (c + self.smoothing * marg) / (c.sum(1, keepdims=True) + self.smoothing)
        return np.log(p).astype(np.float32)


class GeoVidScorer:
    """Гео-приор и совпадение вида услуги как батчевые матрицы [batch, n_items]."""

    def __init__(self, q_df, items_df, geo: GeoPrior):
        self.q_locs_u, q_loc_idx = np.unique(q_df['search_location_id'].values, return_inverse=True)
        self.i_locs_u, i_loc_idx = np.unique(items_df['item_location_id'].values, return_inverse=True)
        self.geo_lp = geo.logp_matrix(self.q_locs_u, self.i_locs_u)
        self.q_loc_idx, self.i_loc_idx = q_loc_idx, i_loc_idx
        q_vid, i_vid = q_df['search_vid'].values, items_df['item_vid'].values
        vocab = {v: k for k, v in enumerate(sorted(set(i_vid) | set(q_vid)))}
        self.q_vid = np.array([vocab[v] if v else -1 for v in q_vid])
        self.i_vid = np.array([vocab[v] if v else -2 for v in i_vid])

    def batch_components(self, sl):
        geo = self.geo_lp[self.q_loc_idx[sl]][:, self.i_loc_idx]
        vid = (self.q_vid[sl, None] == self.i_vid[None, :]).astype(np.float32)
        return geo, vid


def recall_at_k(pred_idx, positives_idx, k=50):
    """Recall@k: |топ-k ∩ релевантные| / |релевантные|, усреднённая по запросам."""
    r = [len(set(p[:k]) & pos) / len(pos) for p, pos in zip(pred_idx, positives_idx)]
    return float(np.mean(r)), np.array(r)

# %% [markdown]
# ## 4. BM25 по леммам
#
# Своя реализация на sparse-матрицах (без внешней библиотеки BM25): вес ненулевого вхождения —
# `tf·(k1+1) / (tf + k1·(1-b+b·dl/avgdl))`, IDF применяется со стороны запроса. Скор пары
# запрос-документ — скалярное произведение, поэтому батч запросов считается одним матричным умножением.
# Лемматизация — `pymorphy3` с кэшем на уникальное слово (словоформ на порядок меньше, чем вхождений).

# %%
import pymorphy3
from sklearn.feature_extraction.text import CountVectorizer

_STOPWORDS = {
    'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'к', 'у', 'о', 'же', 'ли', 'а', 'но', 'что', 'как',
    'это', 'том', 'при', 'за', 'над', 'под', 'без', 'между', 'или', 'чтобы', 'также', 'еще', 'ещё', 'уже',
    'очень', 'не', 'ни', 'так', 'то', 'все', 'всё', 'его', 'её', 'их', 'он', 'она', 'они', 'мы', 'вы', 'я',
    'быть', 'был', 'была', 'были', 'есть', 'нет', 'да', 'ну', 'вот', 'там', 'тут', 'здесь', 'можно', 'нужно',
}
_TOKEN_RE = re.compile(r'[а-яёa-z0-9]+', re.IGNORECASE)
_morph = pymorphy3.MorphAnalyzer()
_lemma_cache = {}


def _lemma(word):
    lemma = _lemma_cache.get(word)
    if lemma is None:
        lemma = word if word.isdigit() else _morph.parse(word)[0].normal_form
        _lemma_cache[word] = lemma
    return lemma


def tokenize_lemmatize(text):
    words = _TOKEN_RE.findall(text.lower())
    return [_lemma(w) for w in words if w not in _STOPWORDS and len(w) > 1]


class BM25Index:
    def __init__(self, texts, k1=1.5, b=0.75, max_df=0.3, min_df=1):
        self.vectorizer = CountVectorizer(tokenizer=tokenize_lemmatize, preprocessor=lambda x: x,
                                          token_pattern=None, max_df=max_df, min_df=min_df)
        counts = self.vectorizer.fit_transform(texts).tocsr()
        n_docs = counts.shape[0]
        dl = np.asarray(counts.sum(axis=1)).ravel()
        avgdl = max(dl.mean(), 1e-6)
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

        tf = counts.data.astype(np.float32)
        dl_per_nnz = np.repeat(dl, np.diff(counts.indptr))
        denom = tf + k1 * (1 - b + b * dl_per_nnz / avgdl)
        weighted = counts.astype(np.float32).copy()
        weighted.data = tf * (k1 + 1) / denom
        self.doc_matrix = weighted.tocsr()
        self.n_docs = n_docs

    def query_matrix(self, texts):
        counts = self.vectorizer.transform(texts).astype(np.float32)
        return (counts @ sp.diags(self.idf)).tocsr()

    def score_batch(self, query_sparse_batch):
        return np.asarray((self.doc_matrix @ query_sparse_batch.T).todense()).T

# %% [markdown]
# ## 5. Модель эмбеддингов
#
# Базовая модель — `deepvk/USER-base`, e5-подобная (обучена с префиксами `query: `/`passage: `). У
# `sentence-transformers==6.1.0` два нюанса, из-за которых модель без патчей не загружается и работает
# некорректно:
#
# - при пустом `path` у модуля `Normalize` в `modules.json` загрузчик по ошибке передаёт в `Normalize()`
#   конфиг корневой архитектуры трансформера вместо собственных параметров модуля;
# - у модели задан `default_prompt_name='query'`: без явного `prompt=''` в `encode()` библиотека сама
#   добавляет `query: ` ко всем текстам, включая объявления.

# %%
from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modules.normalize import Normalize


@classmethod
def _patched_normalize_load(cls, model_name_or_path='', subfolder='', token=None, cache_folder=None,
                            revision=None, local_files_only=False, **kwargs):
    if not model_name_or_path:
        return cls()
    config = cls.load_config(model_name_or_path=model_name_or_path, subfolder=subfolder, token=token,
                             cache_folder=cache_folder, revision=revision, local_files_only=local_files_only)
    return cls(**{k: v for k, v in config.items() if k in cls.config_keys})


Normalize.load = _patched_normalize_load

MAX_SEQ_LEN = 192


def load_model(name_or_path):
    m = SentenceTransformer(str(name_or_path), device=DEVICE)
    m.max_seq_length = MAX_SEQ_LEN
    return m


@torch.no_grad()
def encode(model, texts, bs=256):
    if DEVICE == 'cuda':
        model.half()
    emb = model.encode(texts, batch_size=bs, convert_to_tensor=True, normalize_embeddings=True,
                       prompt='', show_progress_bar=True)
    model.float()
    return emb.float().cpu().numpy()


model = load_model('deepvk/USER-base' if TRAIN_MODEL else WEIGHTS_PATH)

# %% [markdown]
# ## 6. Дообучение
#
# Контрастивное дообучение на парах «текст запроса → текст выбранного объявления» из `fit_rows`:
# `CachedMultipleNegativesRankingLoss` берёт остальные объявления батча как негативы, `NO_DUPLICATES`
# не даёт одинаковым текстам запроса попасть в один батч (иначе частый запрос получил бы свой же
# правильный ответ в качестве негатива). Обучение в fp32: `DeBERTa` (архитектура `USER-base`) под
# автокастом смешанной точности переполняет attention-маску при приведении диапазона fp32 к fp16.

# %%
if TRAIN_MODEL:
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer import losses
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    from transformers import TrainerCallback

    class TimeBudget(TrainerCallback):
        """Ограничение по времени обучения — контрастивное дообучение на полном train занимает
        несколько часов, явный предел защищает от непредсказуемо долгой сессии."""

        def __init__(self, hours):
            self.deadline = time.time() + hours * 3600

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() > self.deadline:
                control.should_training_stop = True

    ft_pairs = fit_rows[['query_text', 'item_text']].rename(columns={'query_text': 'anchor',
                                                                      'item_text': 'positive'})
    ft_pairs = ft_pairs.drop_duplicates().sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    ft_dataset = Dataset.from_pandas(ft_pairs, preserve_index=False)

    ft_loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=16)
    ft_args = SentenceTransformerTrainingArguments(
        output_dir=str(OUT_DIR / 'ckpt'),
        num_train_epochs=2,
        per_device_train_batch_size=64,
        learning_rate=2e-5,
        warmup_steps=0.05,
        fp16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=50,
        save_strategy='no',
        report_to='none',
        seed=SEED,
    )
    ft_trainer = SentenceTransformerTrainer(model=model, args=ft_args, train_dataset=ft_dataset, loss=ft_loss,
                                            callbacks=[TimeBudget(hours=4.0)])
    ft_trainer.train()
    model.save(str(WEIGHTS_PATH))
    del ft_trainer, ft_loss, ft_dataset, ft_pairs
    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown]
# ## 7. Гео/вид-буст для отбора кандидатов
#
# BM25 и эмбеддинги отбирают top-K по сырому скору с добавкой `alpha·log P(локация) +
# beta·[вид услуги совпал]` — без неё объявления из целевого города вытесняются похожими объявлениями
# из других регионов и не доходят до этапа ранжирования. Вес подбирается отдельно для каждого источника
# (у них разный масштаб скора — BM25 не ограничен сверху, косинус в [-1, 1]) на подвыборке валидации
# в масштабе, близком к бенчмарку, максимизацией recall кандидатов в top-K.

# %%
K_BM25, K_EMBED = 150, 150
GEO_ALPHAS = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
VID_BETAS = [0, 0.02, 0.05, 0.1, 0.2, 0.4]
BM25_GEO_ALPHAS = [0, 0.5, 1, 2, 4, 8, 16, 32]
BM25_VID_BETAS = [0, 1, 2, 4, 8, 16]
TUNE_SAMPLE_SIZE = 600

bm25_val = BM25Index(val_corpus.item_text.tolist())
item_emb_val = encode(model, val_corpus.item_text.tolist())
geo_val = GeoPrior(fit_rows.search_location_id.values, fit_rows.item_location_id.values)
scorer_val = GeoVidScorer(val_q, val_corpus, geo_val)
val_item_pos = {iid: i for i, iid in enumerate(val_corpus.item_id)}
val_positives_idx = [set(val_item_pos[i] for i in pos) for pos in val_q.positives]


def tune_retrieval_boost(query_texts, geo_vid_scorer, dense_score_fn, positives_idx, k,
                         alpha_grid, beta_grid, batch=150):
    """Подбирает alpha/beta, добавляемые к сырому скору источника перед отбором top-k, по recall
    кандидатов в top-k. dense_score_fn(sl) -> [b, n_items] сырой скор для батча query_texts[sl]."""
    raw_chunks, geo_chunks, vid_chunks = [], [], []
    for st in range(0, len(query_texts), batch):
        sl = slice(st, st + batch)
        raw_chunks.append(dense_score_fn(sl))
        geo, vid = geo_vid_scorer.batch_components(sl)
        geo_chunks.append(geo)
        vid_chunks.append(vid)
    raw, geo, vid = np.concatenate(raw_chunks), np.concatenate(geo_chunks), np.concatenate(vid_chunks)
    n_items = raw.shape[1]

    best = (0.0, 0.0, -1.0)
    for a in alpha_grid:
        for b in beta_grid:
            s = raw + a * geo + b * vid
            kk = min(k, n_items)
            idx = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
            hits = [len(set(row) & pos) / len(pos) if pos else 1.0 for row, pos in zip(idx, positives_idx)]
            recall = float(np.mean(hits))
            if recall > best[2]:
                best = (a, b, recall)
    return best


rng = np.random.default_rng(SEED)
tune_idx = rng.choice(len(val_q), size=min(TUNE_SAMPLE_SIZE, len(val_q)), replace=False)
tune_texts = val_q.query_text.values[tune_idx].tolist()
tune_positives = [val_positives_idx[i] for i in tune_idx]
tune_scorer = GeoVidScorer(val_q.iloc[tune_idx].reset_index(drop=True), val_corpus, geo_val)
tune_q_emb = encode(model, tune_texts)

alpha_bm25, beta_bm25, _ = tune_retrieval_boost(
    tune_texts, tune_scorer, lambda sl: bm25_val.score_batch(bm25_val.query_matrix(tune_texts[sl])),
    tune_positives, K_BM25, BM25_GEO_ALPHAS, BM25_VID_BETAS)
alpha_embed, beta_embed, _ = tune_retrieval_boost(
    tune_texts, tune_scorer, lambda sl: tune_q_emb[sl] @ item_emb_val.T,
    tune_positives, K_EMBED, GEO_ALPHAS, VID_BETAS)
print(f'boost: bm25 alpha={alpha_bm25} beta={beta_bm25}, embed alpha={alpha_embed} beta={beta_embed}')

del tune_q_emb, tune_scorer
gc.collect()

# %% [markdown]
# ## 8. Пул кандидатов и признаки для LightGBM
#
# top-K по BM25 и по эмбеддингам (отбор — с гео/вид-бустом) объединяются без повторов. Оба сырых скора
# считаются батчами запросов как плотные матрицы `[batch, n_items]`, что даёт точный скор и ранг для
# любого кандидата объединения без повторных вычислений. В признаки идут сырые значения BM25/эмбеддингов
# без буста — гео-приор и вид услуги уже есть отдельными признаками.

# %%
FEATURE_COLS = ['bm25_score', 'bm25_rank', 'embed_score', 'embed_rank', 'geo_logprob', 'vid_match',
                'geo_x_vid', 'both_sources',
                'item_rating', 'item_reviews_log', 'item_price_log', 'item_phone_hidden', 'item_msg_forbidden']


def build_pool_features(query_texts, item_emb, item_df, bm25_index, geo_vid_scorer,
                        alpha_bm25, beta_bm25, alpha_embed, beta_embed, positives_idx=None, batch=150):
    """Признаки по парам (запрос, кандидат). positives_idx — множества позиций релевантных
    объявлений на запрос; если переданы, добавляется label и считается потолок recall пула."""
    q_emb = encode(model, query_texts)
    item_rating = item_df['item_rating'].fillna(0).values.astype(np.float32)
    item_reviews = np.log1p(item_df['item_rating_reviews_count'].fillna(0).values).astype(np.float32)
    item_price = np.log1p(item_df['item_price'].fillna(0).astype(float).clip(lower=0).values).astype(np.float32)
    item_phone_hidden = item_df['item_is_phone_hidden'].astype(np.float32).values
    item_msg_forbidden = item_df['item_is_message_forbidden'].astype(np.float32).values
    n_items = len(item_df)

    rows = []
    ceiling_hits, ceiling_total = 0, 0
    for st in range(0, len(query_texts), batch):
        sl = slice(st, st + batch)
        qb = q_emb[sl]
        embed_dense = qb @ item_emb.T
        bm25_dense = bm25_index.score_batch(bm25_index.query_matrix(query_texts[sl]))
        geo_dense, vid_dense = geo_vid_scorer.batch_components(sl)
        bm25_select = bm25_dense + alpha_bm25 * geo_dense + beta_bm25 * vid_dense
        embed_select = embed_dense + alpha_embed * geo_dense + beta_embed * vid_dense

        for i in range(qb.shape[0]):
            qpos = st + i
            top_bm25 = np.argpartition(-bm25_select[i], min(K_BM25, n_items - 1))[:K_BM25]
            top_embed = np.argpartition(-embed_select[i], min(K_EMBED, n_items - 1))[:K_EMBED]
            union = np.union1d(top_bm25, top_embed)

            rank_bm25 = np.full(len(union), K_BM25 + 1, dtype=np.int32)
            pos_in_union = {v: j for j, v in enumerate(union)}
            for r, idx in enumerate(top_bm25[np.argsort(-bm25_select[i, top_bm25])], start=1):
                rank_bm25[pos_in_union[idx]] = r
            rank_embed = np.full(len(union), K_EMBED + 1, dtype=np.int32)
            for r, idx in enumerate(top_embed[np.argsort(-embed_select[i, top_embed])], start=1):
                rank_embed[pos_in_union[idx]] = r

            geo_vals, vid_vals = geo_dense[i, union], vid_dense[i, union]
            df = pd.DataFrame({
                'qpos': qpos, 'item_pos': union,
                'bm25_score': bm25_dense[i, union], 'bm25_rank': rank_bm25,
                'embed_score': embed_dense[i, union], 'embed_rank': rank_embed,
                'geo_logprob': geo_vals, 'vid_match': vid_vals,
                'geo_x_vid': geo_vals * vid_vals,
                'both_sources': ((rank_bm25 <= K_BM25) & (rank_embed <= K_EMBED)).astype(np.float32),
                'item_rating': item_rating[union], 'item_reviews_log': item_reviews[union],
                'item_price_log': item_price[union],
                'item_phone_hidden': item_phone_hidden[union], 'item_msg_forbidden': item_msg_forbidden[union],
            })
            if positives_idx is not None:
                pos = positives_idx[qpos]
                df['label'] = df.item_pos.isin(pos).astype(np.int32)
                ceiling_hits += len(pos & set(union))
                ceiling_total += len(pos)
            rows.append(df)

    pool = pd.concat(rows, ignore_index=True)
    ceiling = ceiling_hits / ceiling_total if positives_idx is not None and ceiling_total else None
    return pool, ceiling

# %% [markdown]
# ## 9. Обучение LightGBM
#
# Обучающий пул строится по 30 000 поисков train и корпусу того же масштаба, что бенчмарк (189 212
# объявлений: реальные позитивы + случайные дистракторы) — это устраняет расхождение в масштабе BM25
# IDF и скоров между обучением ранжирования и инференсом. 10% запросов пула отделяется для ранней
# остановки; локальная валидация (раздел 2) в обучении не участвует.

# %%
N_TRAIN_QUERIES = 30_000

LGBM_PARAMS = dict(objective='lambdarank', metric='ndcg', eval_at=[50], boosting_type='gbdt',
                   num_leaves=31, learning_rate=0.03, min_data_in_leaf=50,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                   n_estimators=1200, seed=SEED, verbosity=-1)


def sample_corpus(items_df, keep_item_ids, corpus_size, seed):
    """corpus_size объявлений: сначала item_id из keep_item_ids, затем случайные дистракторы."""
    keep_rows = items_df[items_df.item_id.isin(keep_item_ids)]
    pool = items_df[~items_df.item_id.isin(keep_item_ids)]
    n_distr = max(0, corpus_size - len(keep_rows))
    distr = pool.sample(n=min(n_distr, len(pool)), random_state=seed)
    return pd.concat([keep_rows, distr], ignore_index=True)


train_groups = (fit_rows.groupby('gid', sort=False)
                .agg(search_location_id=('search_location_id', 'first'), search_vid=('search_vid', 'first'),
                     query_text=('query_text', 'first'), positives=('item_id', lambda x: sorted(set(x))))
                .reset_index())
train_groups = train_groups.sample(n=min(N_TRAIN_QUERIES, len(train_groups)), random_state=SEED
                                   ).reset_index(drop=True)

needed_positive_ids = set().union(*train_groups.positives)
train_items = sample_corpus(train.drop_duplicates('item_id')[[c for c in train.columns if c.startswith('item_')]],
                            needed_positive_ids, CORPUS_SIZE, SEED)

item_pos_train = {iid: i for i, iid in enumerate(train_items.item_id)}
train_positives_idx = [set(item_pos_train[i] for i in pos if i in item_pos_train) for pos in train_groups.positives]
keep = [len(p) > 0 for p in train_positives_idx]
train_groups = train_groups[keep].reset_index(drop=True)
train_positives_idx = [p for p, k in zip(train_positives_idx, keep) if k]

bm25_train = BM25Index(train_items.item_text.tolist())
item_emb_train = encode(model, train_items.item_text.tolist())
geo_train = GeoPrior(fit_rows.search_location_id.values, fit_rows.item_location_id.values)
scorer_train = GeoVidScorer(train_groups, train_items, geo_train)

train_pool, train_ceiling = build_pool_features(
    train_groups.query_text.tolist(), item_emb_train, train_items, bm25_train, scorer_train,
    alpha_bm25, beta_bm25, alpha_embed, beta_embed, positives_idx=train_positives_idx)
print(f'train pool: {len(train_pool)} rows, candidate ceiling recall={train_ceiling:.4f}')

# %%
import lightgbm as lgb

uniq_qpos = train_pool.qpos.unique()
dev_qpos = set(rng.choice(uniq_qpos, size=max(1, int(0.1 * len(uniq_qpos))), replace=False))


def to_lgbm_dataset(pool, qpos_subset):
    part = pool[pool.qpos.isin(qpos_subset)].sort_values('qpos').reset_index(drop=True)
    group_sizes = part.groupby('qpos', sort=False).size().values
    return lgb.Dataset(part[FEATURE_COLS], label=part.label, group=group_sizes)


fit_ds = to_lgbm_dataset(train_pool, set(uniq_qpos) - dev_qpos)
dev_ds = to_lgbm_dataset(train_pool, dev_qpos)

gbm = lgb.train(LGBM_PARAMS, fit_ds, valid_sets=[dev_ds],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])
print(pd.Series(gbm.feature_importance(importance_type='gain'), index=FEATURE_COLS)
      .sort_values(ascending=False).to_string())

del train_pool, fit_ds, dev_ds, item_emb_train
gc.collect()

# %% [markdown]
# ## 10. Оценка на локальной валидации и разбор ошибок
#
# Потолок пула — recall кандидатов до ранжирования; итоговый Recall@50 — после LightGBM. Разница между
# ними — это то, что теряет именно ранжирование, а не генерация кандидатов. Запросы с недостающим
# объявлением делятся на два типа: объявление вообще не попало в пул (не хватило BM25/эмбеддингов) или
# попало, но не удержалось в топ-50 (не хватило признаков/весов ранжирования).

# %%
val_pool, val_ceiling = build_pool_features(
    val_q.query_text.tolist(), item_emb_val, val_corpus, bm25_val, scorer_val,
    alpha_bm25, beta_bm25, alpha_embed, beta_embed, positives_idx=val_positives_idx)

val_pool['score'] = gbm.predict(val_pool[FEATURE_COLS])
val_top50 = (val_pool.sort_values(['qpos', 'score'], ascending=[True, False])
            .groupby('qpos')['item_pos'].apply(lambda s: s.head(50).tolist())
            .reindex(range(len(val_q)), fill_value=[]))

final_recall, per_q = recall_at_k(val_top50.tolist(), val_positives_idx, k=50)
print(f'candidate ceiling: {val_ceiling:.4f}, final Recall@50: {final_recall:.4f}')

val_pool_hit = (val_pool.groupby('qpos')
               .apply(lambda g: bool(set(g.item_pos) & val_positives_idx[g.name]), include_groups=False))
miss_stage1 = [i for i in range(len(val_q)) if val_positives_idx[i] and not val_pool_hit.get(i, False)]
miss_stage2 = [i for i in range(len(val_q)) if per_q[i] < 1.0 and i not in miss_stage1 and val_positives_idx[i]]
print(f'потеряно на генерации кандидатов: {len(miss_stage1)}, на ранжировании: {len(miss_stage2)}')

del val_pool, item_emb_val
gc.collect()

# %% [markdown]
# ## 11. Ответ для бенчмарка

# %%
bm25_bench = BM25Index(bench_items.item_text.tolist())
item_emb_bench = encode(model, bench_items.item_text.tolist())
geo_bench = GeoPrior(train.search_location_id.values, train.item_location_id.values)
scorer_bench = GeoVidScorer(bench_q, bench_items, geo_bench)

bench_pool, _ = build_pool_features(
    bench_q.query_text.tolist(), item_emb_bench, bench_items, bm25_bench, scorer_bench,
    alpha_bm25, beta_bm25, alpha_embed, beta_embed)
bench_pool['score'] = gbm.predict(bench_pool[FEATURE_COLS])
bench_top50 = (bench_pool.sort_values(['qpos', 'score'], ascending=[True, False])
              .groupby('qpos')['item_pos'].apply(lambda s: s.head(50).tolist())
              .reindex(range(len(bench_q)), fill_value=[]))

item_ids = bench_items.item_id.values
answer = pd.DataFrame({'query_id': bench_q.query_id.values,
                       'answer': [' '.join(item_ids[idx]) for idx in bench_top50.tolist()]})

assert answer.query_id.is_unique and set(answer.query_id) == set(bench_q.query_id)
id_set = set(item_ids)
for a in answer.answer:
    ids = a.split(' ') if a else []
    assert len(ids) <= 50 and len(set(ids)) == len(ids) and all(len(i) == 16 and i in id_set for i in ids)

answer.to_csv(OUT_DIR / 'answer.csv', index=False)
gbm.save_model(str(OUT_DIR / 'lgbm_model.txt'))
print(answer.head())
