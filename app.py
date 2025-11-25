# app.py
import os
import re
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ==================== НАСТРОЙКИ ====================

DEFAULT_SNAP_DIR = os.getenv(
    "YT_RADAR_DATA_DIR",
    "snapshots_raw",  # папка внутри проекта, рядом с app.py
)
# сколько часов считаем видео "свежим" по умолчанию
DEFAULT_FRESH_HOURS = 72.0

st.set_page_config(
    page_title="YouTube Category Radar",
    layout="wide",
)

st.title("YouTube Radar")

st.markdown(
    """
Приложение смотрит на трендовые видео YouTube (по 150 из каждой категории в сегменте ru).

Анализ проводится в двух уровнях:

1. **Анализ внутри одного снапшота**  
2. **Динамика между снапшотами** — сравнение двух: последнего и предпоследнего, поиск роста/падения.
"""
)

# ==================== ВСТАВКА С ФОРМУЛАМИ И ПРИМЕРАМИ ====================

with st.expander("Расчёты и какие формулы с простыми примерами"):
    st.markdown(
        r"""
### 1. Что такое «снапшот»

**Снапшот** — это слепок трендов в конкретный момент времени.  

Дополнительно рассчитываются:

- `snapshot_ts` — точное время снимка (например, 2025-11-20 10:00:00).  
- `snapshot_date`, `snapshot_time` — дата и время отдельно.  

---

### 2. Возраст видео и скорость за жизнь ролика

YouTube API даёт:

- `views` — сколько просмотров уже набрало видео;  
- `published_at` — когда ролик впервые вышел.

Рассчитываем **возраст ролика**:

\[
age\_hours = \frac{snapshot\_ts - published\_at}{3600 \text{ секунд}}
\]

и примерную **среднюю скорость за всю жизнь**:

\[
views\_{per\_hour} \approx \frac{views}{age\_hours}
\]

#### 💡 Пример

- Ролик вышел вчера в 10:00.  
- Сейчас мы делаем снимок сегодня в 10:00.  
- Значит, ролик живёт уже **24 часа**.  
- У него 240 000 просмотров.

Тогда:

- `age_hours = 24`  
- `views_per_hour ≈ 240000 / 24 = 10 000 просмотров в час`

Это примерная оценка, насколько ролик вообще «быстрый» по жизни.

---

### 3. «Свежее» видео (fresh_hours)

Мы вводим порог, например:

- `fresh_hours = 72` → всё, что младше 72 часов, считаем свежим.

Условие свежести:

\[
age\_hours \leq fresh\_hours
\]

Если условие выполняется → `is_fresh = True`, иначе → `False`.

#### 💡 Пример

- Видео №1: живёт 10 часов → свежее.  
- Видео №2: живёт 50 часов → свежее.  
- Видео №3: живёт 120 часов → **уже не свежее** при `fresh_hours = 72`.

---

### 4. Метрики по категориям в одном снапшоте

Берём одну категорию (например, Music) и смотрим на все её видео в этом снимке.

Для категории считаем:

1. **Объём просмотров**:

\[
volume = \sum\_{video \in category} views\_{video}
\]

2. **Общая скорость всех видео**:

\[
velocity\_{total} = \sum\_{video \in category} views\_{per\_hour,video}
\]

3. **Сколько видео свежие**:

\[
fresh\_videos = \#\{ video \in category \mid age\_hours \leq fresh\_hours \}
\]

4. **Сколько всего видео** — `videos_cnt`.  

5. **Доля свежих видео**:

\[
freshness = \frac{fresh\_videos}{videos\_cnt}
\]

6. **Скорость только свежих видео**:

\[
fresh\_velocity =
\sum\_{video \in category,\, age\_hours \leq fresh\_hours}
views\_{per\_hour,video}
\]

#### 💡 Пример по категории

Пусть в категории «Music» в этом снапшоте 3 ролика:

| Видео | Просмотры (views) | Скорость (views_per_hour) | Возраст (часов) |
|-------|-------------------|---------------------------|-----------------|
| A     | 200 000           | 10 000                    | 20              |
| B     | 100 000           | 5 000                     | 10              |
| C     | 50 000            | 1 000                     | 200             |

При `fresh_hours = 72` → свежие только A и B (20 и 10 часов).

Тогда:

- **volume** = 200 000 + 100 000 + 50 000 = **350 000**  
- **velocity_total** = 10 000 + 5 000 + 1 000 = **16 000**  
- **videos_cnt** = 3  
- **fresh_videos** = 2 (A и B)  
- **freshness** = 2 / 3 ≈ 0.67 (2/3 видео свежие)  
- **fresh_velocity** = 10 000 + 5 000 = **15 000** (C старый, мы его тут не учитываем)

---

### 5. Доли категорий от общего пирога

Сначала считаем **сумму по всем категориям**, например:

\[
total\_volume = \sum\_{cat} volume\_{cat}
\]
\[
total\_fresh\_velocity = \sum\_{cat} fresh\_velocity\_{cat}
\]

Потом доли:

\[
volume\_share = \frac{volume}{total\_volume}
\]
\[
fresh\_velocity\_share = \frac{fresh\_velocity}{total\_fresh\_velocity}
\]

#### 💡 Пример с двумя категориями

Пусть:

- **Music**: `volume = 350 000`, `fresh_velocity = 15 000`  
- **Gaming**: `volume = 150 000`, `fresh_velocity = 5 000`

Тогда:

- `total_volume = 350 000 + 150 000 = 500 000`  
- `total_fresh_velocity = 15 000 + 5 000 = 20 000`

Доли:

- `volume_share(Music) = 350000 / 500000 = 0.7` → **70% всех просмотров**  
- `fresh_velocity_share(Music) = 15000 / 20000 = 0.75` → **75% скорости новых видео**

Вывод: Music не просто крупнее, но и свежее/быстрее по новинкам.

---

### 6. Метрики по темам (тегам) в одном снапшоте

Берём одну категорию (например, Music) и внутри неё смотрим **теги** (темы).  
Для каждого тега:

1. **Объём просмотров по тегу**:

\[
volume\_{tag} = \sum\_{video \in tag} views\_{video}
\]

2. **Общая скорость по тегу**:

\[
velocity\_{total,tag} = \sum\_{video \in tag} views\_{per\_hour,video}
\]

3. **Скорость только по свежим видео с тегом**:

\[
velocity\_{tag} =
\sum\_{video \in tag,\, age\_hours \leq fresh\_hours}
views\_{per\_hour,video}
\]

4. **videos_cnt** — сколько уникальных роликов с этим тегом.  
5. **fresh_videos** — сколько из них свежие.  
6. **freshness** — доля свежих:

\[
freshness\_{tag} = \frac{fresh\_videos\_{tag}}{videos\_cnt\_{tag}}
\]

#### 💡 Пример по тегу

Пусть в категории Music есть тег `#covers`, и с ним 3 свежих видео и 1 старое:

| Видео | Просмотры | Скорость | Возраст (часов) | Свежий? |
|-------|-----------|---------|------------------|---------|
| A     | 50 000    | 5 000   | 5                | да      |
| B     | 30 000    | 3 000   | 10               | да      |
| C     | 20 000    | 2 000   | 50               | да      |
| D     | 40 000    | 500     | 150              | нет     |

Тогда:

- **volume_tag** = 50k + 30k + 20k + 40k = **140 000**  
- **velocity_total_tag** = 5k + 3k + 2k + 0.5k = **10 500**  
- **velocity_tag** = только свежие: 5k + 3k + 2k = **10 000**  
- **videos_cnt** = 4  
- **fresh_videos** = 3  
- **freshness_tag** = 3 / 4 = 0.75

Вывод: тема `#covers` живёт хорошо: высокий объём, высокая скорость, много свежих роликов.

---

### 7. Динамика категорий между двумя снапшотами

Берём **две точки во времени**:  
например, вчера 10:00 (`t1`) и сегодня 10:00 (`t2`).

Для категории считаем:

- `volume_t1`, `volume_t2` — объёмы просмотров;  
- `fresh_velocity_t1`, `fresh_velocity_t2` — скорость новых видео;  
- `freshness_t1`, `freshness_t2` — доля свежих.

Дальше **дельты**:

\[
volume\_delta = volume\_{t2} - volume\_{t1}
\]

\[
fresh\_velocity\_delta = fresh\_velocity\_{t2} - fresh\_velocity\_{t1}
\]

\[
freshness\_delta = freshness\_{t2} - freshness\_{t1}
\]

#### 💡 Пример

Музыка вчера и сегодня:

- Вчера: `volume_t1 = 300 000`, `fresh_velocity_t1 = 10 000`, `freshness_t1 = 0.5`  
- Сегодня: `volume_t2 = 450 000`, `fresh_velocity_t2 = 18 000`, `freshness_t2 = 0.7`

Тогда:

- `volume_delta = 450k − 300k = +150 000`  
- `fresh_velocity_delta = 18k − 10k = +8 000`  
- `freshness_delta = 0.7 − 0.5 = +0.2`

Вывод: и просмотры, и скорость, и доля свежих сильно выросли → категория реально разогрелась.

---

### 8. Динамика тем (тегов) между двумя снапшотами

Берём тег, который есть **и в t1, и в t2**.

По нему считаем:

\[
volume\_delta = volume\_{t2} - volume\_{t1}
\]
\[
velocity\_delta = velocity\_{t2} - velocity\_{t1}
\]
\[
freshness\_delta = freshness\_{t2} - freshness\_{t1}
\]

#### 💡 Пример для тега

Тег `#ai` в категории Science:

- Вчера: `volume_t1 = 100 000`, `velocity_t1 = 4 000`, `freshness_t1 = 0.4`  
- Сегодня: `volume_t2 = 200 000`, `velocity_t2 = 9 000`, `freshness_t2 = 0.7`

Дельты:

- `volume_delta = +100 000`  
- `velocity_delta = +5 000`  
- `freshness_delta = +0.3`

То есть не просто стало больше просмотров, но и **сильно выросла скорость** и доля свежих видео — тема «взрывается».

---

### 9. Динамика отдельных видео между двумя снапшотами

Для каждого видео, которое есть **в обоих** снапшотах, берём:

- `views_t1`, `views_t2` — просмотры «до» и «после».  
- Время между снимками:

\[
hours\_between\_snaps = \frac{ts2 - ts1}{3600 \text{ секунд}}
\]

и считаем:

\[
views\_delta = views\_{t2} - views\_{t1}
\]

\[
views\_{per\_hour\_between} =
\frac{views\_delta}{hours\_between\_snaps}
\]

#### 💡 Пример для видео

- Вчера в 10:00 (`t1`): у ролика 50 000 просмотров.  
- Сегодня в 10:00 (`t2`): уже 110 000 просмотров.  
- Между снимками прошло 24 часа.

Тогда:

- `views_delta = 110000 − 50000 = 60 000`  
- `hours_between_snaps = 24`  
- `views_per_hour_between = 60000 / 24 ≈ 2 500 просмотров в час`

То есть **за последние сутки** ролик ехал со скоростью ~2.5k просмотров в час — именно это мы и показываем в динамике видео.
"""
    )

# ==================== ШПАРГАЛКА ПО ИНТЕРПРЕТАЦИЯМ ====================

with st.expander("Быстрая шпаргалка: как читать метрики без формул"):
    st.markdown(
        """
### Категории (страница «Обзор категорий»)

- **volume**  
  Чем больше → тем больше суммарных просмотров у категории.  
  *Вопрос:* «Кто сейчас собирает больше всего просмотров вообще?»

- **velocity_total**  
  Чем больше → тем выше общий темп набора просмотров у всех видео категории (и старых, и новых).  
  *Вопрос:* «Где просмотры текут быстрее всего, если смотреть на всю массу роликов?»

- **fresh_velocity**  
  Чем больше → тем быстрее растут **новые** ролики (младше `fresh_hours`).  
  *Вопрос:* «Где сейчас вспыхивают самые живые новинки?»

- **freshness** (0–1)  
  0.2 → мало свежих видео, 0.8 → почти все свежие.  
  *Вопрос:* «Эта категория живёт за счёт старых хитов или там много свежего контента?»

- **volume_share / velocity_share / fresh_velocity_share**  
  Это «кусок пирога» в процентах.  
  *volume_share* — сколько % всех просмотров приходится на категорию.  
  *fresh_velocity_share* — сколько % скорости новых видео она забирает.  
  Если доля по fresh_velocity выше, чем доля по volume → категория **перегрета новинками**.

---

### Темы (теги) в категории

- **volume (тега)**  
  Большой объём → тема уже «толстая», вокруг неё накопилось много просмотров.

- **velocity (тега)**  
  Смотрим только на свежие ролики.  
  Большая скорость → по этой теме новинки прямо сейчас хорошо заходят.

- **freshness (тега)**  
  Большая доля свежих роликов → тема живая, туда регулярно что-то выпускают.  
  Низкая → тема держится на старых видео.

- **status (Trending / Emerging / Mature / Declining / Frozen / Other)**  
  Это короткий вердикт по теме:
  - **Trending** — сейчас очень быстро растёт, много свежих просмотров.  
  - **Emerging** — только набирает силу: скорости уже высокие, но объёмы ещё не огромные.  
  - **Mature** — большой трафик, стабильный темп, «крупная, устоявшаяся» тема.  
  - **Declining** — когда-то была большой, но скорость и свежесть падают.  
  - **Frozen** — мало и новых видео, и скорости, трафик оживает редко.  
  - **Other** — всё, что не попало в явные паттерны.

---

### Динамика категорий и тем (две точки во времени)

- **volume_delta**  
  > 0 → просмотров стало больше, категория / тема растёт в абсолюте.  
  < 0 → просмотров стало меньше (перестали смотреть или много видео выпало из топа).

- **fresh_velocity_delta** (для категорий)  
  > 0 → новинки в этой категории набирают просмотры быстрее, чем раньше.  
  < 0 → новинки замедлились, хайп подустал.

- **velocity_delta** (для тегов)  
  > 0 → тема в свежих видео разгоняется, растёт интерес.  
  < 0 → свежие видео по теме стали меньше смотреть.

- **freshness_delta**  
  > 0 → доля свежих видео выросла, авторы принесли много нового контента.  
  < 0 → новых роликов стало меньше, трафик держится на старом.

---

### Динамика отдельных видео

- **views_delta**  
  Просто разница просмотров между двумя снапшотами.  
  Большое значение → ролик хорошо крутился в выбранном окне.

- **views_per_hour_between**  
  Самая понятная штука: *«как быстро росли просмотры именно между этими двумя точками»*.  
  Условно:  
  - 100–500 в час → мелкое движение;  
  - 1 000–5 000 в час → довольно бодро;  
  - 10 000+ в час → очень сильный рост на этом промежутке.

Суть:

- Смотри **категории** → где вообще сейчас концентрируется трафик и новинки.  
- Смотри **темы** → какие сюжеты тащат рост внутри выбранной категории.  
- Смотри **видео** → какие конкретные ролики делают этот рост.
"""
    )

# ==================== ОЧИСТКА ТЕГОВ ====================

STOP_TAGS = {
    # общие англ.
    "short", "shorts", "youtubeshorts",
    "viral", "trend", "trending",
    "fyp", "foryou", "reels",
    "subscribe", "subscribenow", "sub",
    "like", "likes", "likethis",
    "follow", "followme",
    "new", "news", "newvideo", "video", "videos",
    "live", "stream",
    "channel", "official", "tv",

    # русские служебные
    "шорт", "шортс", "шортсы",
    "тренд", "тренды", "втренде",
    "подписка", "подпишись", "подписаться",
    "лайк", "лайки", "ставьлайк",
    "рекомендации", "рекомендацииютуба",
    "новое", "новинка", "новинкавидео", "видео",
    "стрим", "прямойэфир",
    "канал", "официальный",
}

EXTRA_STOP_SUBSTR = (
    "official", "офишл", "офишлканал",
    "channel", "канал",
)


def clean_tag(raw_tag: str):
    """
    Чистим один тег.

    Возвращаем:
      - нормализованный тег (str), если он годится как тема;
      - None, если это мусор, который не хотим видеть как тему.
    """
    if not isinstance(raw_tag, str):
        return None

    tag = raw_tag.strip().lower()

    # убираем служебные символы по краям
    while tag and tag[0] in "#@!*_•.- ":
        tag = tag[1:]
    while tag and tag[-1] in "#@!*_•.- ":
        tag = tag[:-1]

    if not tag:
        return None

    if len(tag) < 2:
        return None

    if tag.isdigit():
        return None

    alnum_count = sum(ch.isalnum() for ch in tag)
    if alnum_count == 0:
        return None
    if alnum_count / len(tag) < 0.4:
        return None

    if tag in STOP_TAGS:
        return None

    for sub in EXTRA_STOP_SUBSTR:
        if sub in tag:
            return None

    return tag


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

FNAME_RE = re.compile(r"ytcat_(?P<cat>\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.csv")

TAG_COLS = [
    "tags_api_raw",
    "hashtags_extracted",
    "tags_common",
    "tags_only_api",
    "tags_only_hash",
]


def parse_snapshot_ts_from_name(filename: str):
    """
    Имя файла вида ytcat_{catid}_{YYYYMMDD}_{HHMMSS}.csv
    Превращаем дату+время в datetime.
    """
    m = FNAME_RE.match(filename)
    if not m:
        return None
    date_str = m.group("date")
    time_str = m.group("time")
    dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    return dt


def parse_tag_json(s):
    """
    Аккуратно парсим строку с тегами.
    """
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [str(x).strip().lower() for x in val if str(x).strip()]
        return [str(val).strip().lower()]
    except Exception:
        return [s.strip().lower()]


def build_all_tags_uniq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Собираем все теги и хэштеги в одну колонку all_tags_uniq (JSON-строка),
    сразу с очисткой.
    """

    def merge_row(row):
        tags_set = set()

        for col in TAG_COLS:
            if col not in row:
                continue
            raw_list = parse_tag_json(row[col])
            for raw_tag in raw_list:
                cleaned = clean_tag(raw_tag)
                if cleaned:
                    tags_set.add(cleaned)

        return json.dumps(sorted(tags_set), ensure_ascii=False)

    df = df.copy()
    df["all_tags_uniq"] = df.apply(merge_row, axis=1)
    return df


@st.cache_data(show_spinner=True)
def load_snapshots_from_directory(directory: str) -> pd.DataFrame:
    """
    Читаем все CSV-файлы вида ytcat_*.csv из указанной папки.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Папка '{directory}' не найдена")

    dfs = []
    for fname in os.listdir(directory):
        if not fname.endswith(".csv"):
            continue

        snap_ts = parse_snapshot_ts_from_name(fname)
        if snap_ts is None:
            continue

        fpath = os.path.join(directory, fname)
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"Не удалось прочитать {fpath}: {e}")
            continue

        df["snapshot_file"] = fname
        df["snapshot_ts"] = snap_ts

        if "category_id" not in df.columns:
            m = FNAME_RE.match(fname)
            if m:
                df["category_id"] = m.group("cat")

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    full = pd.concat(dfs, ignore_index=True)
    full["snapshot_ts"] = pd.to_datetime(full["snapshot_ts"])
    full["snapshot_date"] = full["snapshot_ts"].dt.date
    full["snapshot_time"] = full["snapshot_ts"].dt.time

    if "category_id" in full.columns:
        full["category_id"] = full["category_id"].astype(str)

    full = build_all_tags_uniq(full)

    return full


def compute_growth_between_snapshots(
    df: pd.DataFrame,
    ts1: datetime,
    ts2: datetime,
) -> pd.DataFrame:
    """
    Сравнение двух снапшотов по video_id.
    """
    df1 = df[df["snapshot_ts"] == ts1].copy()
    df2 = df[df["snapshot_ts"] == ts2].copy()

    if df1.empty or df2.empty:
        return pd.DataFrame()

    base_cols = [
        "video_id",
        "title",
        "channel_title",
        "category_id",
        "category_name",
        "views",
        "views_per_hour",
        "from_shorts",
        "duration_sec",
        "all_tags_uniq",
        "published_at",
    ]

    for c in base_cols:
        if c not in df1.columns:
            df1[c] = None
        if c not in df2.columns:
            df2[c] = None

    df1 = df1[base_cols].rename(
        columns={c: f"{c}_t1" for c in base_cols if c != "video_id"}
    )
    df2 = df2[base_cols].rename(
        columns={c: f"{c}_t2" for c in base_cols if c != "video_id"}
    )

    merged = df1.merge(df2, on="video_id", how="inner")
    if merged.empty:
        return merged

    hours_diff = (ts2 - ts1).total_seconds() / 3600.0
    if hours_diff <= 0:
        hours_diff = 1e-6

    merged["hours_between_snaps"] = hours_diff
    merged["views_delta"] = merged["views_t2"] - merged["views_t1"]
    merged["views_per_hour_between"] = merged["views_delta"] / hours_diff

    merged = merged.sort_values("views_per_hour_between", ascending=False)
    return merged


def compute_category_metrics_for_snapshot(
    df: pd.DataFrame,
    snapshot_ts: datetime,
    fresh_hours: float = DEFAULT_FRESH_HOURS,
) -> pd.DataFrame:
    """
    Метрики по категориям для одного снапшота.
    """
    df2 = df[df["snapshot_ts"] == snapshot_ts].copy()
    if df2.empty:
        return pd.DataFrame()

    df2["views"] = pd.to_numeric(df2.get("views", 0), errors="coerce").fillna(0)
    df2["views_per_hour"] = pd.to_numeric(
        df2.get("views_per_hour", 0.0), errors="coerce"
    ).fillna(0.0)

    if "published_at" in df2.columns:
        df2["published_at_dt"] = pd.to_datetime(
            df2["published_at"], errors="coerce", utc=True
        ).dt.tz_convert(None)
        df2["age_hours"] = (
            df2["snapshot_ts"] - df2["published_at_dt"]
        ).dt.total_seconds() / 3600.0
    else:
        df2["age_hours"] = np.nan

    df2["is_fresh"] = df2["age_hours"] <= fresh_hours

    if "category_name" not in df2.columns:
        df2["category_name"] = df2["category_id"].astype(str)
    df2["category_label"] = df2["category_name"].fillna(df2["category_id"].astype(str))

    rows = []
    for (cat_id, cat_name), g in df2.groupby(["category_id", "category_label"]):
        volume = g["views"].sum()
        velocity_total = g["views_per_hour"].sum()
        fresh_velocity = g.loc[g["is_fresh"], "views_per_hour"].sum()
        videos_cnt = g["video_id"].nunique()
        fresh_videos = int(g["is_fresh"].sum())
        freshness = fresh_videos / videos_cnt if videos_cnt > 0 else 0.0

        rows.append(
            {
                "category_id": str(cat_id),
                "category_name": cat_name,
                "volume": volume,
                "velocity_total": velocity_total,
                "fresh_velocity": fresh_velocity,
                "videos_cnt": videos_cnt,
                "fresh_videos": fresh_videos,
                "freshness": freshness,
            }
        )

    cat_df = pd.DataFrame(rows)
    if cat_df.empty:
        return cat_df

    total_volume = cat_df["volume"].sum() or 1e-6
    total_velocity = cat_df["velocity_total"].sum() or 1e-6
    total_fresh_velocity = cat_df["fresh_velocity"].sum() or 1e-6

    cat_df["volume_share"] = cat_df["volume"] / total_volume
    cat_df["velocity_share"] = cat_df["velocity_total"] / total_velocity
    cat_df["fresh_velocity_share"] = cat_df["fresh_velocity"] / total_fresh_velocity

    return cat_df


def compute_tag_metrics_for_df_slice(
    df_slice: pd.DataFrame,
    fresh_hours: float = DEFAULT_FRESH_HOURS,
    min_videos_per_tag: int = 1,
) -> pd.DataFrame:
    """
    Метрики по тегам для одного снапшота и одной категории.
    """
    df2 = df_slice.copy()
    if df2.empty:
        return pd.DataFrame()

    df2["views"] = pd.to_numeric(df2.get("views", 0), errors="coerce").fillna(0)
    df2["views_per_hour"] = pd.to_numeric(
        df2.get("views_per_hour", 0.0), errors="coerce"
    ).fillna(0.0)

    if "published_at" in df2.columns:
        df2["published_at_dt"] = pd.to_datetime(
            df2["published_at"], errors="coerce", utc=True
        ).dt.tz_convert(None)
        if "snapshot_ts" in df2.columns:
            snap_ts = df2["snapshot_ts"].iloc[0]
        else:
            snap_ts = datetime.now()
        df2["age_hours"] = (
            pd.to_datetime(snap_ts) - df2["published_at_dt"]
        ).dt.total_seconds() / 3600.0
    else:
        df2["age_hours"] = np.nan

    df2["is_fresh"] = df2["age_hours"] <= fresh_hours

    tag_rows = []
    for _, row in df2.iterrows():
        tags = parse_tag_json(row.get("all_tags_uniq", "[]"))
        if not tags:
            continue
        v_views = row["views"]
        v_vel = row["views_per_hour"]
        v_fresh = bool(row["is_fresh"])
        vid = row["video_id"]
        for t in tags:
            tag_rows.append(
                {
                    "tag": t,
                    "video_id": vid,
                    "views": v_views,
                    "velocity_total": v_vel,
                    "velocity_fresh": v_vel if v_fresh else 0.0,
                    "is_fresh": v_fresh,
                }
            )

    if not tag_rows:
        return pd.DataFrame()

    tag_df = pd.DataFrame(tag_rows)

    tag_agg = (
        tag_df.groupby("tag")
        .agg(
            volume=("views", "sum"),
            velocity_total=("velocity_total", "sum"),
            velocity=("velocity_fresh", "sum"),
            videos_cnt=("video_id", "nunique"),
            fresh_videos=("is_fresh", "sum"),
        )
        .reset_index()
    )

    tag_agg["freshness"] = tag_agg["fresh_videos"] / tag_agg["videos_cnt"]

    tag_agg = tag_agg[tag_agg["videos_cnt"] >= min_videos_per_tag].copy()
    if tag_agg.empty:
        return tag_agg

    p75_velocity = float(tag_agg["velocity"].quantile(0.75))
    p90_velocity = float(tag_agg["velocity"].quantile(0.90))
    p75_volume = float(tag_agg["volume"].quantile(0.75))
    median_volume = float(tag_agg["volume"].median())
    median_velocity = float(tag_agg["velocity"].median())

    lower_mature_vel = 0.8 * p75_velocity
    upper_mature_vel = 1.2 * p75_velocity

    tag_agg["status"] = "Other"

    trending_mask = (tag_agg["velocity"] >= p90_velocity) & (
        tag_agg["freshness"] > 0.5
    )
    tag_agg.loc[trending_mask, "status"] = "Trending"

    emerging_mask = (
        (tag_agg["status"] == "Other")
        & (tag_agg["velocity"] >= p75_velocity)
        & (tag_agg["volume"] < median_volume)
        & (tag_agg["freshness"] > 0.5)
    )
    tag_agg.loc[emerging_mask, "status"] = "Emerging"

    declining_mask = (
        (tag_agg["status"] == "Other")
        & (tag_agg["volume"] >= p75_volume)
        & (tag_agg["velocity"] < median_velocity)
        & (tag_agg["freshness"] < 0.3)
    )
    tag_agg.loc[declining_mask, "status"] = "Declining"

    mature_mask = (
        (tag_agg["status"] == "Other")
        & (tag_agg["volume"] >= p75_volume)
        & (tag_agg["velocity"] >= lower_mature_vel)
        & (tag_agg["velocity"] <= upper_mature_vel)
    )
    tag_agg.loc[mature_mask, "status"] = "Mature"

    frozen_mask = (
        (tag_agg["status"] == "Other")
        & (tag_agg["volume"] < median_volume)
        & (tag_agg["velocity"] < median_velocity)
    )
    tag_agg.loc[frozen_mask, "status"] = "Frozen"

    return tag_agg


def explode_tags_for_growth(df_growth: pd.DataFrame) -> pd.DataFrame:
    """
    Берём таблицу с ростом видео между снапшотами и смотрим,
    какие теги набрали больше всего дополнительных просмотров.
    """
    if df_growth.empty:
        return pd.DataFrame()

    rows = []
    for _, row in df_growth.iterrows():
        tags = parse_tag_json(row.get("all_tags_uniq_t2", "[]"))
        delta = row.get("views_delta", 0)
        for t in tags:
            if t:
                rows.append({"tag": t, "views_delta": delta})

    if not rows:
        return pd.DataFrame()

    tag_df = pd.DataFrame(rows)
    agg = (
        tag_df.groupby("tag", as_index=False)["views_delta"]
        .sum()
        .sort_values("views_delta", ascending=False)
    )
    return agg


# ==================== ЗАГРУЗКА ДАННЫХ ====================

st.sidebar.header("Папка со снапшотами")

snap_dir_input = st.sidebar.text_input(
    "Путь к папке со снапшотами",
    value=DEFAULT_SNAP_DIR,
    help="Все файлы вида ytcat_XXX_YYYYMMDD_HHMMSS.csv должны лежать в этой папке.",
)

if not snap_dir_input:
    st.stop()

try:
    full_df = load_snapshots_from_directory(snap_dir_input)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if full_df.empty:
    st.error("В папке нет валидных снапшотов (ytcat_*.csv).")
    st.stop()

if "category_name" not in full_df.columns:
    full_df["category_name"] = full_df["category_id"].astype(str)

st.success(
    f"Считано {len(full_df)} строк, "
    f"{full_df['snapshot_ts'].nunique()} снапшотов, "
    f"{full_df.get('category_id', pd.Series()).nunique()} категорий."
)

with st.expander("Список снапшотов по датам"):
    snap_summary = (
        full_df.groupby("snapshot_ts")
        .agg(
            videos=("video_id", "nunique"),
            categories=("category_id", "nunique"),
        )
        .reset_index()
    )
    snap_summary["snapshot_ts"] = snap_summary["snapshot_ts"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    st.table(snap_summary)

snapshots = sorted(full_df["snapshot_ts"].dropna().unique())
if len(snapshots) < 1:
    st.error("Нет ни одного снапшота.")
    st.stop()

snap_labels = {ts: ts.strftime("%Y-%m-%d %H:%M:%S") for ts in snapshots}

# индексы для "последний vs предпоследний" по умолчанию
last_idx = len(snapshots) - 1
prev_idx = max(0, last_idx - 1)

# ==================== ВЫБОР РЕЖИМА ====================

st.sidebar.header("Режим анализа")

page = st.sidebar.radio(
    "Выбери, что смотреть:",
    options=["Аналитика одного снапшота", "Динамика между снапшотами"],
)

# ===================================================================
#                 СТРАНИЦА 1. АНАЛИТИКА ОДНОГО СНАПШОТА
# ===================================================================

if page == "Аналитика одного снапшота":
    st.subheader("Аналитика одного снапшота")

    tab_cat, tab_tags, tab_videos = st.tabs(
        ["Обзор категорий", "Темы внутри категории", "Видео внутри категории"]
    )

    # ------------------ Вкладка: Обзор категорий ------------------
    with tab_cat:
        st.markdown(
            """
Здесь мы смотрим на картину по категориям в один момент времени.

- Снапшот — слепок трендов (конкретная дата и время).
- Для каждой категории считаем объём просмотров и скорость, особенно по новым роликам.
"""
        )

        col_settings = st.columns(2)
        with col_settings[0]:
            ts_one = st.selectbox(
                "Выбери снапшот (момент времени)",
                options=snapshots,
                index=last_idx,
                format_func=lambda x: snap_labels[x],
                key="one_ts_cat",
            )
        with col_settings[1]:
            fresh_hours_one = st.number_input(
                "Сколько часов считаем видео новым (fresh_hours)",
                min_value=1.0,
                max_value=168.0,
                value=DEFAULT_FRESH_HOURS,
                step=1.0,
                key="one_fresh_cat",
            )

        cat_metrics = compute_category_metrics_for_snapshot(
            full_df, snapshot_ts=ts_one, fresh_hours=fresh_hours_one
        )

        if cat_metrics.empty:
            st.warning("Для выбранного снапшота нет данных по категориям.")
        else:
            st.markdown(f"Снапшот: **{snap_labels[ts_one]}**")

            col_stats = st.columns(3)
            with col_stats[0]:
                st.metric("Категорий", len(cat_metrics))
            with col_stats[1]:
                st.metric("Суммарные просмотры", f"{cat_metrics['volume'].sum():.0f}")
            with col_stats[2]:
                st.metric(
                    "Суммарная скорость новых видео",
                    f"{cat_metrics['fresh_velocity'].sum():.0f}",
                )

            metric_for_share = st.selectbox(
                "По какой метрике рисовать доли категорий",
                options=["volume_share", "velocity_share", "fresh_velocity_share"],
                format_func=lambda x: {
                    "volume_share": "Доля по просмотрам (volume_share)",
                    "velocity_share": "Доля по общей скорости (velocity_share)",
                    "fresh_velocity_share": "Доля по скорости новых видео (fresh_velocity_share)",
                }[x],
                key="one_metric_share",
            )

            plot_df = cat_metrics.copy().sort_values(
                metric_for_share, ascending=False
            )

            chart = (
                alt.Chart(plot_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        f"{metric_for_share}:Q",
                        title={
                            "volume_share": "Доля просмотров от всех категорий",
                            "velocity_share": "Доля скорости от всех категорий",
                            "fresh_velocity_share": "Доля скорости новых видео",
                        }[metric_for_share],
                        axis=alt.Axis(format="~%"),
                    ),
                    y=alt.Y(
                        "category_name:N",
                        sort="-x",
                        title="Категория",
                    ),
                    tooltip=[
                        "category_id:N",
                        "category_name:N",
                        "videos_cnt:Q",
                        "volume:Q",
                        "velocity_total:Q",
                        "fresh_velocity:Q",
                        "freshness:Q",
                        "volume_share:Q",
                        "velocity_share:Q",
                        "fresh_velocity_share:Q",
                    ],
                )
                .properties(height=500)
            )

            st.altair_chart(chart, use_container_width=True)

            with st.expander("Таблица по категориям и объяснение колонок"):
                st.dataframe(cat_metrics, use_container_width=True)

                st.markdown("### Обозначения колонок (категории, один снапшот)")

                st.markdown(
                    "- **category_id** — ID категории YouTube.\n"
                    "- **category_name** — название категории.\n"
                    "- **videos_cnt** — число уникальных видео в категории."
                )

                st.markdown("#### Объёмы и скорости")

                st.markdown("**volume** — суммарные просмотры категории:")
                st.latex(
                    r"volume = \sum_{\text{video} \in \text{category}} "
                    r"views_{\text{video}}"
                )

                st.markdown("**velocity_total** — суммарная скорость всех видео:")
                st.latex(
                    r"velocity_{\text{total}} = "
                    r"\sum_{\text{video} \in \text{category}} "
                    r"views_{\text{per\_hour, video}}"
                )

                st.markdown("**fresh_videos** — сколько видео в категории свежие:")
                st.latex(
                    r"fresh_{\text{videos}} = "
                    r"\#\{\text{video} \in \text{category} \mid "
                    r"age_{\text{hours}} \leq fresh_{\text{hours}}\}"
                )

                st.markdown("**freshness** — доля свежих видео:")
                st.latex(
                    r"freshness = "
                    r"\frac{fresh_{\text{videos}}}{videos_{\text{cnt}}}"
                )

                st.markdown(
                    "**fresh_velocity** — суммарная скорость только свежих видео:"
                )
                st.latex(
                    r"fresh_{\text{velocity}} = "
                    r"\sum_{\text{video} \in \text{category},\, "
                    r"age_{\text{hours}} \leq fresh_{\text{hours}}} "
                    r"views_{\text{per\_hour, video}}"
                )

                st.markdown("#### Доли по отношению ко всем категориям")

                st.markdown("Сначала считаем суммы по всем категориям:")
                st.latex(
                    r"total_{\text{volume}} = "
                    r"\sum_{\text{cat}} volume_{\text{cat}}"
                )
                st.latex(
                    r"total_{\text{velocity}} = "
                    r"\sum_{\text{cat}} velocity_{\text{total, cat}}"
                )
                st.latex(
                    r"total_{\text{fresh\_velocity}} = "
                    r"\sum_{\text{cat}} fresh_{\text{velocity, cat}}"
                )

                st.markdown("Потом доли:")

                st.markdown("**volume_share** — доля просмотров категории:")
                st.latex(
                    r"volume_{\text{share}} = "
                    r"\frac{volume}{total_{\text{volume}}}"
                )

                st.markdown("**velocity_share** — доля общей скорости:")
                st.latex(
                    r"velocity_{\text{share}} = "
                    r"\frac{velocity_{\text{total}}}{total_{\text{velocity}}}"
                )

                st.markdown("**fresh_velocity_share** — доля скорости новых видео:")
                st.latex(
                    r"fresh_{\text{velocity\_share}} = "
                    r"\frac{fresh_{\text{velocity}}}{total_{\text{fresh\_velocity}}}"
                )

    # ------------------ Вкладка: Темы внутри категории ------------------
    with tab_tags:
        st.markdown(
            """
Здесь мы смотрим на темы (теги) внутри одной категории в один момент времени.

Мы считаем, сколько просмотров у темы и как быстро растут новые видео с этим тегом.
"""
        )

        col_settings = st.columns(3)
        with col_settings[0]:
            ts_tags = st.selectbox(
                "Снапшот для анализа тем",
                options=snapshots,
                index=last_idx,
                format_func=lambda x: snap_labels[x],
                key="one_ts_tags",
            )

        df_for_ts = full_df[full_df["snapshot_ts"] == ts_tags].copy()
        df_for_ts["category_label"] = df_for_ts["category_name"].fillna(
            df_for_ts["category_id"].astype(str)
        )

        available_categories = (
            df_for_ts[["category_id", "category_label"]]
            .drop_duplicates()
            .sort_values("category_label")
        )

        if available_categories.empty:
            st.warning("Для выбранного снапшота нет категорий.")
        else:
            cat_options = [
                f"{row.category_label} (id={row.category_id})"
                for row in available_categories.itertuples(index=False)
            ]
            cat_map = {
                f"{row.category_label} (id={row.category_id})": (
                    row.category_id,
                    row.category_label,
                )
                for row in available_categories.itertuples(index=False)
            }

            with col_settings[1]:
                selected_cat_option = st.selectbox(
                    "Категория",
                    options=cat_options,
                    index=0,
                    key="one_cat_tags",
                )
            selected_cat_id, selected_cat_label = cat_map[selected_cat_option]

            with col_settings[2]:
                fresh_hours_tags = st.number_input(
                    "Сколько часов считаем видео новым",
                    min_value=1.0,
                    max_value=168.0,
                    value=DEFAULT_FRESH_HOURS,
                    step=1.0,
                    key="one_fresh_tags",
                )

            min_videos_per_tag = st.number_input(
                "Минимальное число видео с тегом",
                min_value=1,
                max_value=50,
                value=2,
                step=1,
                key="one_min_videos_tag",
            )

            df_slice = df_for_ts[df_for_ts["category_id"] == str(selected_cat_id)].copy()
            tag_metrics = compute_tag_metrics_for_df_slice(
                df_slice,
                fresh_hours=fresh_hours_tags,
                min_videos_per_tag=min_videos_per_tag,
            )

            if tag_metrics.empty:
                st.warning("Для этой категории и настроек нет данных по тегам.")
            else:
                st.markdown(
                    f"Снапшот: **{snap_labels[ts_tags]}**, "
                    f"категория: **{selected_cat_label} (id={selected_cat_id})**"
                )

                col_stats = st.columns(3)
                with col_stats[0]:
                    st.metric("Тегов всего", len(tag_metrics))
                with col_stats[1]:
                    st.metric(
                        "Медианный объём (просмотры)",
                        f"{tag_metrics['volume'].median():.0f}",
                    )
                with col_stats[2]:
                    st.metric(
                        "Медианная скорость новых видео",
                        f"{tag_metrics['velocity'].median():.0f}",
                    )

                st.subheader("Карта тем: объём против скорости новых видео")

                scatter_df = tag_metrics.copy()
                scatter_df["status_cat"] = scatter_df["status"].astype("category")

                chart_tags = (
                    alt.Chart(scatter_df)
                    .mark_circle()
                    .encode(
                        x=alt.X(
                            "volume:Q",
                            title="Объём темы (просмотры)",
                        ),
                        y=alt.Y(
                            "velocity:Q",
                            title="Скорость новых видео (сумма views/hour)",
                        ),
                        size=alt.Size(
                            "videos_cnt:Q",
                            title="Количество видео",
                            scale=alt.Scale(range=[30, 400]),
                        ),
                        color=alt.Color(
                            "status_cat:N",
                            title="Статус темы",
                        ),
                        tooltip=[
                            "tag:N",
                            "status:N",
                            "volume:Q",
                            "velocity:Q",
                            "velocity_total:Q",
                            "videos_cnt:Q",
                            "freshness:Q",
                        ],
                    )
                    .properties(height=500)
                    .interactive()
                )

                st.altair_chart(chart_tags, use_container_width=True)

                st.subheader("Топ тем по скорости новых видео")
                top_tags_by_vel = tag_metrics.sort_values(
                    "velocity", ascending=False
                ).head(50)
                st.dataframe(top_tags_by_vel, use_container_width=True)

                with st.expander("Объяснение колонок для тем"):
                    st.markdown("### Что означают столбцы в таблице по темам (тегам)")

                    st.markdown(
                        "- **tag** — сама тема/тег.\n"
                        "- **videos_cnt** — сколько уникальных видео используют этот тег."
                    )

                    st.markdown("#### Объёмы и скорости")

                    st.markdown(
                        "**volume** — суммарные просмотры всех видео с этим тегом:"
                    )
                    st.latex(
                        r"volume_{\text{tag}} = "
                        r"\sum_{\text{video} \in \text{tag}} views_{\text{video}}"
                    )

                    st.markdown(
                        "**velocity_total** — суммарная скорость всех видео с тегом:"
                    )
                    st.latex(
                        r"velocity_{\text{total, tag}} = "
                        r"\sum_{\text{video} \in \text{tag}} "
                        r"views_{\text{per\_hour, video}}"
                    )

                    st.markdown(
                        "**velocity** — суммарная скорость только свежих видео с тегом:"
                    )
                    st.latex(
                        r"velocity_{\text{tag}} = "
                        r"\sum_{\text{video} \in \text{tag},\, "
                        r"age_{\text{hours}} \leq fresh_{\text{hours}}} "
                        r"views_{\text{per\_hour, video}}"
                    )

                    st.markdown(
                        "**fresh_videos** — сколько видео с тегом являются свежими."
                    )
                    st.markdown("**freshness** — доля свежих видео:")
                    st.latex(
                        r"freshness_{\text{tag}} = "
                        r"\frac{fresh_{\text{videos, tag}}}{videos_{\text{cnt, tag}}}"
                    )

                    st.markdown("#### Статус темы (`status`)")

                    st.markdown(
                        """
- **Trending** — сейчас очень быстро растёт, много свежих просмотров.  
- **Emerging** — только набирает силу: скорости уже высокие, но объёмы ещё не огромные.  
- **Mature** — большой трафик, стабильный темп, «крупная, устоявшаяся» тема.  
- **Declining** — когда-то была большой, но скорость и свежесть падают.  
- **Frozen** — мало и новых видео, и скорости, трафик оживает редко.  
- **Other** — всё, что не попало в явные паттерны.

Статус назначается по положению темы относительно перцентилей `velocity` и `volume` внутри категории.
"""
                    )

                st.subheader("Разбивка по статусам")
                status_order = [
                    "Trending",
                    "Emerging",
                    "Mature",
                    "Declining",
                    "Frozen",
                    "Other",
                ]
                for st_name in status_order:
                    subset = tag_metrics[tag_metrics["status"] == st_name]
                    if subset.empty:
                        continue
                    with st.expander(f"{st_name} — {len(subset)} тегов"):
                        st.dataframe(
                            subset.sort_values("velocity", ascending=False),
                            use_container_width=True,
                        )

                with st.expander("Сырые данные по тегам"):
                    st.dataframe(tag_metrics, use_container_width=True)

    # ------------------ Вкладка: Видео внутри категории ------------------
    with tab_videos:
        st.markdown(
            """
Здесь мы смотрим на отдельные видео внутри категории в один момент (без динамики).
"""
        )

        col_settings = st.columns(2)
        with col_settings[0]:
            ts_vid = st.selectbox(
                "Снапшот для просмотра видео",
                options=snapshots,
                index=last_idx,
                format_func=lambda x: snap_labels[x],
                key="one_ts_videos",
            )

        df_ts = full_df[full_df["snapshot_ts"] == ts_vid].copy()
        df_ts["category_label"] = df_ts["category_name"].fillna(
            df_ts["category_id"].astype(str)
        )
        available_categories_v = (
            df_ts[["category_id", "category_label"]]
            .drop_duplicates()
            .sort_values("category_label")
        )

        if available_categories_v.empty:
            st.warning("Для выбранного снапшота нет категорий.")
        else:
            cat_options_v = [
                f"{row.category_label} (id={row.category_id})"
                for row in available_categories_v.itertuples(index=False)
            ]
            cat_map_v = {
                f"{row.category_label} (id={row.category_id})": (
                    row.category_id,
                    row.category_label,
                )
                for row in available_categories_v.itertuples(index=False)
            }

            with col_settings[1]:
                selected_cat_option_v = st.selectbox(
                    "Категория",
                    options=cat_options_v,
                    index=0,
                    key="one_cat_videos",
                )
            selected_cat_id_v, selected_cat_label_v = cat_map_v[selected_cat_option_v]

            df_cat_vid = df_ts[df_ts["category_id"] == str(selected_cat_id_v)].copy()
            if df_cat_vid.empty:
                st.warning("В этой категории нет видео для выбранного снапшота.")
            else:
                st.markdown(
                    f"Снапшот: **{snap_labels[ts_vid]}**, "
                    f"категория: **{selected_cat_label_v} (id={selected_cat_id_v})**"
                )

                shorts_filter = st.radio(
                    "Фильтр по типу видео",
                    options=["Все", "Только shorts", "Только не shorts"],
                    index=0,
                    key="one_shorts_filter",
                )

                if "from_shorts" in df_cat_vid.columns:
                    if shorts_filter == "Только shorts":
                        df_cat_vid = df_cat_vid[df_cat_vid["from_shorts"] == 1]
                    elif shorts_filter == "Только не shorts":
                        df_cat_vid = df_cat_vid[df_cat_vid["from_shorts"] == 0]

                if df_cat_vid.empty:
                    st.warning("После фильтрации видео не осталось.")
                else:
                    df_cat_vid["views"] = pd.to_numeric(
                        df_cat_vid.get("views", 0), errors="coerce"
                    ).fillna(0)
                    df_cat_vid["views_per_hour"] = pd.to_numeric(
                        df_cat_vid.get("views_per_hour", 0.0), errors="coerce"
                    ).fillna(0.0)

                    top_n_local = st.slider(
                        "Сколько видео показать",
                        min_value=10,
                        max_value=200,
                        value=50,
                        step=10,
                        key="one_top_videos",
                    )

                    def short_title(s, max_len=60):
                        s = str(s)
                        return s if len(s) <= max_len else s[: max_len - 3] + "..."

                    top_videos_cat = df_cat_vid.sort_values(
                        "views_per_hour", ascending=False
                    ).head(top_n_local)
                    top_videos_cat["title_short"] = top_videos_cat["title"].apply(
                        short_title
                    )

                    st.bar_chart(
                        data=top_videos_cat.set_index("title_short")["views_per_hour"]
                    )

                    show_cols = [
                        "video_id",
                        "title",
                        "channel_title",
                        "views",
                        "views_per_hour",
                        "from_shorts",
                        "duration_sec",
                        "published_at",
                    ]
                    show_cols = [c for c in show_cols if c in top_videos_cat.columns]

                    st.dataframe(
                        top_videos_cat[show_cols],
                        use_container_width=True,
                    )

                    with st.expander("Объяснение колонок для видео"):
                        st.markdown(
                            "### Что означают столбцы в таблице по видео (один снапшот)"
                        )

                        st.markdown(
                            "- **video_id** — ID видео.\n"
                            "- **title** — заголовок видео.\n"
                            "- **channel_title** — название канала."
                        )

                        st.markdown("#### Просмотры и скорость")

                        st.markdown("**views** — число просмотров на момент снапшота.")
                        st.markdown(
                            "**views_per_hour** — примерная средняя скорость за жизнь ролика:"
                        )
                        st.latex(
                            r"views_{\text{per\_hour}} \approx "
                            r"\frac{views}{age_{\text{hours}}}"
                        )
                        st.latex(
                            r"age_{\text{hours}} = "
                            r"\frac{snapshot_{\text{ts}} - published_{\text{at}}}{3600}"
                        )

                        st.markdown("#### Про формат")

                        st.markdown(
                            "- **from_shorts** — признак, что видео похоже на Shorts:\n"
                            "  - `1` — шорт (очень короткий или помечен как shorts),\n"
                            "  - `0` — обычный ролик.\n"
                            "- **duration_sec** — длительность видео в секундах.\n"
                            "- **published_at** — дата и время публикации видео."
                        )

# ===================================================================
#                 СТРАНИЦА 2. ДИНАМИКА МЕЖДУ СНАПШОТАМИ
# ===================================================================

elif page == "Динамика между снапшотами":
    st.subheader("Динамика между снапшотами")

    tab_cat_dyn, tab_tags_dyn, tab_videos_dyn = st.tabs(
        ["Категории", "Темы внутри категории", "Видео"]
    )

    # ------------------ ДИНАМИКА КАТЕГОРИЙ ------------------
    with tab_cat_dyn:
        st.markdown(
            """
Здесь мы сравниваем две точки во времени и смотрим, где вырос интерес к новым видео.

По умолчанию сравниваются **последний** снапшот и **предпоследний**.
"""
        )

        col_settings = st.columns(3)
        with col_settings[0]:
            ts1_cat = st.selectbox(
                "Ранний снапшот (было)",
                options=snapshots,
                index=prev_idx,
                format_func=lambda x: snap_labels[x],
                key="dyn_cat_ts1",
            )
        with col_settings[1]:
            ts2_cat = st.selectbox(
                "Поздний снапшот (стало)",
                options=snapshots,
                index=last_idx,
                format_func=lambda x: snap_labels[x],
                key="dyn_cat_ts2",
            )
        with col_settings[2]:
            fresh_hours_dyn_cat = st.number_input(
                "Сколько часов считаем видео новым",
                min_value=1.0,
                max_value=168.0,
                value=DEFAULT_FRESH_HOURS,
                step=1.0,
                key="dyn_cat_fresh",
            )

        if ts2_cat <= ts1_cat:
            st.warning("Поздний снапшот должен быть позже раннего.")
        else:
            cat1 = compute_category_metrics_for_snapshot(
                full_df, ts1_cat, fresh_hours=fresh_hours_dyn_cat
            )
            cat2 = compute_category_metrics_for_snapshot(
                full_df, ts2_cat, fresh_hours=fresh_hours_dyn_cat
            )

            if cat1.empty or cat2.empty:
                st.warning("Не удалось посчитать метрики для одного из снапшотов.")
            else:
                c1 = cat1.add_suffix("_t1")
                c2 = cat2.add_suffix("_t2")

                merged_cat = c1.merge(
                    c2,
                    left_on="category_id_t1",
                    right_on="category_id_t2",
                    how="outer",
                )

                merged_cat["category_id"] = merged_cat["category_id_t1"].fillna(
                    merged_cat["category_id_t2"]
                )
                merged_cat["category_name"] = merged_cat["category_name_t1"].fillna(
                    merged_cat["category_name_t2"]
                )

                for col in [
                    "volume_t1",
                    "volume_t2",
                    "fresh_velocity_t1",
                    "fresh_velocity_t2",
                    "freshness_t1",
                    "freshness_t2",
                ]:
                    if col not in merged_cat.columns:
                        merged_cat[col] = 0.0
                    merged_cat[col] = pd.to_numeric(
                        merged_cat[col], errors="coerce"
                    ).fillna(0.0)

                merged_cat["volume_delta"] = (
                    merged_cat["volume_t2"] - merged_cat["volume_t1"]
                )
                merged_cat["fresh_velocity_delta"] = (
                    merged_cat["fresh_velocity_t2"]
                    - merged_cat["fresh_velocity_t1"]
                )
                merged_cat["freshness_delta"] = (
                    merged_cat["freshness_t2"] - merged_cat["freshness_t1"]
                )

                st.markdown(
                    f"Сравнение: **{snap_labels[ts1_cat]} → {snap_labels[ts2_cat]}**"
                )

                col_stats = st.columns(3)
                with col_stats[0]:
                    st.metric("Категорий", len(merged_cat))
                with col_stats[1]:
                    st.metric(
                        "Суммарная дельта скорости новых видео",
                        f"{merged_cat['fresh_velocity_delta'].sum():.0f}",
                    )
                with col_stats[2]:
                    st.metric(
                        "Категории с ростом Fresh Velocity",
                        int((merged_cat["fresh_velocity_delta"] > 0).sum()),
                    )

                st.subheader("Категории с наибольшим ростом Fresh Velocity")

                top_cat = merged_cat.sort_values(
                    "fresh_velocity_delta", ascending=False
                ).head(20)

                chart_cat = (
                    alt.Chart(top_cat)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "fresh_velocity_delta:Q",
                            title="Прирост скорости новых видео (fresh_velocity_delta)",
                        ),
                        y=alt.Y(
                            "category_name:N",
                            sort="-x",
                            title="Категория",
                        ),
                        tooltip=[
                            "category_id:N",
                            "category_name:N",
                            "volume_t1:Q",
                            "volume_t2:Q",
                            "volume_delta:Q",
                            "fresh_velocity_t1:Q",
                            "fresh_velocity_t2:Q",
                            "fresh_velocity_delta:Q",
                            "freshness_t1:Q",
                            "freshness_t2:Q",
                            "freshness_delta:Q",
                        ],
                    )
                    .properties(height=500)
                )

                st.altair_chart(chart_cat, use_container_width=True)

                with st.expander(
                    "Таблица по категориям (динамика) и пояснение колонок"
                ):
                    show_cols = [
                        "category_id",
                        "category_name",
                        "volume_t1",
                        "volume_t2",
                        "volume_delta",
                        "fresh_velocity_t1",
                        "fresh_velocity_t2",
                        "fresh_velocity_delta",
                        "freshness_t1",
                        "freshness_t2",
                        "freshness_delta",
                    ]
                    st.dataframe(
                        merged_cat[show_cols].sort_values(
                            "fresh_velocity_delta", ascending=False
                        ),
                        use_container_width=True,
                    )

                    st.markdown("### Что означают столбцы в динамике категорий")

                    st.markdown(
                        "**volume_t1 / volume_t2** — объём просмотров в раннем и позднем снапшотах:"
                    )
                    st.latex(r"volume_{t1},\ volume_{t2}")

                    st.markdown("**volume_delta** — изменение объёма:")
                    st.latex(r"volume_{\text{delta}} = volume_{t2} - volume_{t1}")

                    st.markdown(
                        "**fresh_velocity_t1 / fresh_velocity_t2** — скорость новых видео в раннем и позднем снапшоте:"
                    )
                    st.latex(
                        r"fresh_{\text{velocity}, t*} = "
                        r"\sum_{\text{video} \in \text{category},\, "
                        r"age_{\text{hours}} \leq fresh_{\text{hours}}} "
                        r"views_{\text{per\_hour, video}}"
                    )

                    st.markdown(
                        "**fresh_velocity_delta** — изменение скорости новых видео:"
                    )
                    st.latex(
                        r"fresh_{\text{velocity\_delta}} = "
                        r"fresh_{\text{velocity}, t2} - fresh_{\text{velocity}, t1}"
                    )

                    st.markdown(
                        "**freshness_t1 / freshness_t2** — доля свежих видео в категории."
                    )
                    st.markdown("**freshness_delta** — изменение доли свежих видео:")
                    st.latex(
                        r"freshness_{\text{delta}} = "
                        r"freshness_{t2} - freshness_{t1}"
                    )

    # ------------------ ДИНАМИКА ТЕМ ВНУТРИ КАТЕГОРИИ ------------------
    with tab_tags_dyn:
        st.markdown(
            """
Здесь мы смотрим, как меняются темы (теги) внутри одной категории между двумя снапшотами.

По умолчанию — сравнение **последнего** и **предпоследнего** снапшотов.
"""
        )

        col_settings = st.columns(4)
        with col_settings[0]:
            ts1_tags = st.selectbox(
                "Ранний снапшот",
                options=snapshots,
                index=prev_idx,
                format_func=lambda x: snap_labels[x],
                key="dyn_tags_ts1",
            )
        with col_settings[1]:
            ts2_tags = st.selectbox(
                "Поздний снапшот",
                options=snapshots,
                index=last_idx,
                format_func=lambda x: snap_labels[x],
                key="dyn_tags_ts2",
            )

        df_all_cat = full_df.copy()
        df_all_cat["category_label"] = df_all_cat["category_name"].fillna(
            df_all_cat["category_id"].astype(str)
        )
        available_categories_all = (
            df_all_cat[["category_id", "category_label"]]
            .drop_duplicates()
            .sort_values("category_label")
        )
        if available_categories_all.empty:
            st.warning("В данных нет категорий.")
        else:
            cat_options_dyn = [
                f"{row.category_label} (id={row.category_id})"
                for row in available_categories_all.itertuples(index=False)
            ]
            cat_map_dyn = {
                f"{row.category_label} (id={row.category_id})": (
                    row.category_id,
                    row.category_label,
                )
                for row in available_categories_all.itertuples(index=False)
            }

            with col_settings[2]:
                selected_cat_option_dyn = st.selectbox(
                    "Категория",
                    options=cat_options_dyn,
                    index=0,
                    key="dyn_tags_cat",
                )
            selected_cat_id_dyn, selected_cat_label_dyn = cat_map_dyn[
                selected_cat_option_dyn
            ]

            with col_settings[3]:
                fresh_hours_dyn_tags = st.number_input(
                    "Сколько часов считаем видео новым",
                    min_value=1.0,
                    max_value=168.0,
                    value=DEFAULT_FRESH_HOURS,
                    step=1.0,
                    key="dyn_tags_fresh",
                )

            min_videos_per_tag_dyn = st.number_input(
                "Минимальное число видео с тегом",
                min_value=1,
                max_value=50,
                value=2,
                step=1,
                key="dyn_tags_min_videos",
            )

            if ts2_tags <= ts1_tags:
                st.warning("Поздний снапшот должен быть позже раннего.")
            else:
                df_ts1_cat = full_df[
                    (full_df["snapshot_ts"] == ts1_tags)
                    & (full_df["category_id"] == str(selected_cat_id_dyn))
                ].copy()
                df_ts2_cat = full_df[
                    (full_df["snapshot_ts"] == ts2_tags)
                    & (full_df["category_id"] == str(selected_cat_id_dyn))
                ].copy()

                tags_t1 = compute_tag_metrics_for_df_slice(
                    df_ts1_cat,
                    fresh_hours=fresh_hours_dyn_tags,
                    min_videos_per_tag=min_videos_per_tag_dyn,
                )
                tags_t2 = compute_tag_metrics_for_df_slice(
                    df_ts2_cat,
                    fresh_hours=fresh_hours_dyn_tags,
                    min_videos_per_tag=min_videos_per_tag_dyn,
                )

                if tags_t1.empty or tags_t2.empty:
                    st.warning(
                        "Не удалось посчитать метрики по тегам для одного из снапшотов."
                    )
                else:
                    t1 = tags_t1.add_suffix("_t1")
                    t2 = tags_t2.add_suffix("_t2")

                    merged_tags = t1.merge(
                        t2,
                        left_on="tag_t1",
                        right_on="tag_t2",
                        how="inner",
                    )

                    merged_tags["tag"] = merged_tags["tag_t1"]

                    for col in [
                        "volume_t1",
                        "volume_t2",
                        "velocity_t1",
                        "velocity_t2",
                        "freshness_t1",
                        "freshness_t2",
                    ]:
                        if col not in merged_tags.columns:
                            merged_tags[col] = 0.0
                        merged_tags[col] = pd.to_numeric(
                            merged_tags[col], errors="coerce"
                        ).fillna(0.0)

                    merged_tags["volume_delta"] = (
                        merged_tags["volume_t2"] - merged_tags["volume_t1"]
                    )
                    merged_tags["velocity_delta"] = (
                        merged_tags["velocity_t2"] - merged_tags["velocity_t1"]
                    )
                    merged_tags["freshness_delta"] = (
                        merged_tags["freshness_t2"] - merged_tags["freshness_t1"]
                    )

                    st.markdown(
                        f"Категория: **{selected_cat_label_dyn} (id={selected_cat_id_dyn})**  \n"
                        f"Сравнение: **{snap_labels[ts1_tags]} → {snap_labels[ts2_tags]}**"
                    )

                    col_stats = st.columns(3)
                    with col_stats[0]:
                        st.metric("Общих тегов", len(merged_tags))
                    with col_stats[1]:
                        st.metric(
                            "Тегов с ростом скорости",
                            int((merged_tags["velocity_delta"] > 0).sum()),
                        )
                    with col_stats[2]:
                        st.metric(
                            "Тегов с падением скорости",
                            int((merged_tags["velocity_delta"] < 0).sum()),
                        )

                    st.subheader("Теги с наибольшим ростом скорости новых видео")

                    show_cols_tags = [
                        "tag",
                        "volume_t1",
                        "volume_t2",
                        "volume_delta",
                        "velocity_t1",
                        "velocity_t2",
                        "velocity_delta",
                        "freshness_t1",
                        "freshness_t2",
                        "freshness_delta",
                        "status_t1",
                        "status_t2",
                    ]
                    show_cols_tags = [
                        c for c in show_cols_tags if c in merged_tags.columns
                    ]

                    top_tags_dyn_vel = merged_tags.sort_values(
                        "velocity_delta", ascending=False
                    ).head(50)

                    st.dataframe(
                        top_tags_dyn_vel[show_cols_tags],
                        use_container_width=True,
                    )

                    with st.expander("Теги с падением скорости новых видео"):
                        low_tags_dyn_vel = merged_tags.sort_values(
                            "velocity_delta"
                        ).head(50)
                        st.dataframe(
                            low_tags_dyn_vel[show_cols_tags],
                            use_container_width=True,
                        )

                    with st.expander("Объяснение колонок для динамики тем"):
                        st.markdown(
                            "### Что означают столбцы в динамике тем (тегов)"
                        )

                        st.markdown(
                            "**tag** — тема/тег, который присутствует в обоих снапшотах."
                        )

                        st.markdown("#### Объёмы и скорости")

                        st.markdown(
                            "**volume_t1 / volume_t2** — суммарные просмотры темы "
                            "в раннем и позднем снапшотах:"
                        )
                        st.latex(r"volume_{t1},\ volume_{t2}")

                        st.markdown("**volume_delta** — изменение объёма:")
                        st.latex(
                            r"volume_{\text{delta}} = volume_{t2} - volume_{t1}"
                        )

                        st.markdown(
                            "**velocity_t1 / velocity_t2** — скорость новых видео "
                            "по теме в начале и в конце периода:"
                        )
                        st.latex(
                            r"velocity_{t*} = "
                            r"\sum_{\text{video} \in \text{tag},\, "
                            r"age_{\text{hours}} \leq fresh_{\text{hours}}} "
                            r"views_{\text{per\_hour, video}}"
                        )

                        st.markdown(
                            "**velocity_delta** — изменение скорости новых видео:"
                        )
                        st.latex(
                            r"velocity_{\text{delta}} = velocity_{t2} - velocity_{t1}"
                        )

                        st.markdown("#### Свежесть темы")

                        st.markdown(
                            "**freshness_t1 / freshness_t2** — доля свежих видео "
                            "с тегом в каждом снапшоте."
                        )
                        st.markdown(
                            "**freshness_delta** — изменение доли свежих видео:"
                        )
                        st.latex(
                            r"freshness_{\text{delta}} = "
                            r"freshness_{t2} - freshness_{t1}"
                        )

                        st.markdown(
                            "#### Статусы\n\n"
                            "- **status_t1 / status_t2** — статус темы (Trending / Emerging / …) "
                            "в начале и в конце периода."
                        )

    # ------------------ ДИНАМИКА ВИДЕО ------------------
    with tab_videos_dyn:
        st.markdown(
            """
Здесь мы смотрим, как растут отдельные видео между двумя снапшотами.

По умолчанию — сравниваем **последний** и **предпоследний** снепы.
"""
        )

        col_settings = st.columns(3)
        with col_settings[0]:
            ts1_vid = st.selectbox(
                "Ранний снапшот",
                options=snapshots,
                index=prev_idx,
                format_func=lambda x: snap_labels[x],
                key="dyn_vid_ts1",
            )
        with col_settings[1]:
            ts2_vid = st.selectbox(
                "Поздний снапшот",
                options=snapshots,
                index=last_idx,
                format_func=lambda x: snap_labels[x],
                key="dyn_vid_ts2",
            )
        with col_settings[2]:
            min_views_delta_v = st.number_input(
                "Минимальный прирост просмотров",
                min_value=0,
                value=0,
                step=1000,
                key="dyn_vid_min_delta",
            )

        if ts2_vid <= ts1_vid:
            st.warning("Поздний снапшот должен быть позже раннего.")
        else:
            growth_df = compute_growth_between_snapshots(full_df, ts1_vid, ts2_vid)
            if growth_df.empty:
                st.warning("Нет пересечения video_id между выбранными снапшотами.")
            else:
                st.markdown(
                    f"Сравнение: **{snap_labels[ts1_vid]} → {snap_labels[ts2_vid]}**  \n"
                    f"Между снапшотами примерно "
                    f"{(ts2_vid - ts1_vid).total_seconds() / 3600:.1f} часов."
                )

                cat_col_v = (
                    "category_name_t2"
                    if "category_name_t2" in growth_df.columns
                    else "category_id_t2"
                )
                all_cats_v = sorted(growth_df[cat_col_v].dropna().unique())

                shorts_filter_v = st.radio(
                    "Фильтр по типу видео",
                    options=["Все", "Только shorts", "Только не shorts"],
                    index=0,
                    key="dyn_vid_shorts",
                )

                selected_cats_v = st.multiselect(
                    "Категории (по позднему снапшоту)",
                    options=all_cats_v,
                    default=all_cats_v,
                    key="dyn_vid_cats",
                )

                top_n_v = st.slider(
                    "Сколько видео показать",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    key="dyn_vid_top_n",
                )

                filtered_v = growth_df.copy()
                if selected_cats_v:
                    filtered_v = filtered_v[filtered_v[cat_col_v].isin(selected_cats_v)]

                if (
                    shorts_filter_v == "Только shorts"
                    and "from_shorts_t2" in filtered_v.columns
                ):
                    filtered_v = filtered_v[filtered_v["from_shorts_t2"] == 1]
                elif (
                    shorts_filter_v == "Только не shorts"
                    and "from_shorts_t2" in filtered_v.columns
                ):
                    filtered_v = filtered_v[filtered_v["from_shorts_t2"] == 0]

                filtered_v = filtered_v[filtered_v["views_delta"] >= min_views_delta_v]

                if filtered_v.empty:
                    st.warning(
                        "После применения фильтров не осталось видео. Ослабь фильтры."
                    )
                else:
                    col_stats = st.columns(4)
                    with col_stats[0]:
                        st.metric(
                            "Видео (в обоих снапшотах, после фильтров)",
                            len(filtered_v),
                        )
                    with col_stats[1]:
                        st.metric(
                            "Средний прирост просмотров",
                            f"{filtered_v['views_delta'].mean():.0f}",
                        )
                    with col_stats[2]:
                        st.metric(
                            "Медианный прирост просмотров",
                            f"{filtered_v['views_delta'].median():.0f}",
                        )
                    with col_stats[3]:
                        st.metric(
                            "Макс. скорость роста (views/час)",
                            f"{filtered_v['views_per_hour_between'].max():.0f}",
                        )

                    def short_title_dyn(s, max_len=60):
                        s = str(s)
                        return s if len(s) <= max_len else s[: max_len - 3] + "..."

                    top_videos = filtered_v.sort_values(
                        "views_per_hour_between", ascending=False
                    ).head(top_n_v)
                    top_videos_display = top_videos.copy()
                    top_videos_display["title_short"] = top_videos_display[
                        "title_t2"
                    ].apply(short_title_dyn)

                    st.bar_chart(
                        data=top_videos_display.set_index("title_short")[
                            "views_per_hour_between"
                        ]
                    )

                    show_cols_v = [
                        "video_id",
                        "title_t2",
                        "channel_title_t2",
                        cat_col_v,
                        "views_t1",
                        "views_t2",
                        "views_delta",
                        "views_per_hour_between",
                        "from_shorts_t2",
                        "duration_sec_t2",
                        "published_at_t2",
                    ]
                    show_cols_v = [
                        c for c in show_cols_v if c in top_videos_display.columns
                    ]

                    st.dataframe(
                        top_videos_display[show_cols_v],
                        use_container_width=True,
                    )

                    with st.expander("Объяснение колонок для динамики видео"):
                        st.markdown("### Что означают столбцы в динамике видео")

                        st.markdown(
                            "**views_t1 / views_t2** — просмотры видео "
                            "в раннем и позднем снапшотах."
                        )
                        st.markdown("**views_delta** — прирост просмотров:")
                        st.latex(
                            r"views_{\text{delta}} = views_{t2} - views_{t1}"
                        )

                        st.markdown(
                            "**hours_between_snaps** — число часов между снапшотами:"
                        )
                        st.latex(
                            r"hours_{\text{between\_snaps}} = "
                            r"\frac{ts_{2} - ts_{1}}{3600}"
                        )

                        st.markdown(
                            "**views_per_hour_between** — средняя скорость роста "
                            "просмотров именно в этом окне:"
                        )
                        st.latex(
                            r"views_{\text{per\_hour\_between}} = "
                            r"\frac{views_{\text{delta}}}{hours_{\text{between\_snaps}}}"
                        )

                        st.markdown(
                            "Эта метрика показывает, как быстро видео набирало просмотры "
                            "за выбранный промежуток, независимо от общего возраста."
                        )

                    st.subheader("Теги по росту просмотров в этом окне")

                    tag_growth_v = explode_tags_for_growth(filtered_v)
                    if tag_growth_v.empty:
                        st.info("Не удалось собрать теги для выбранного набора видео.")
                    else:
                        top_tags_v = tag_growth_v.head(30)
                        st.bar_chart(
                            data=top_tags_v.set_index("tag")["views_delta"]
                        )
                        st.dataframe(top_tags_v, use_container_width=True)

                    with st.expander("Сырые строки по видео"):
                        st.dataframe(filtered_v, use_container_width=True)
