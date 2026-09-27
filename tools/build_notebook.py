"""Збирає notebooks/hw3_nyc_taxi.ipynb."""
import json
import pathlib

import nbformat as nbf

ROOT = pathlib.Path(__file__).resolve().parents[1]
INTERP = json.loads((ROOT / "tools/interpretations.json").read_text(encoding="utf-8"))

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))


def query(sql_text):
    code(f'q("""\n{sql_text.strip()}\n""")')


md("""
# ДЗ 3. NYC Yellow Taxi: завантаження, типізація, аудит якості та EDA у PostgreSQL

**Студентка:** Samoilenko Mariia

**Датасет:** NYC TLC Trip Record Data — Yellow Taxi, січень 2024
([офіційна сторінка](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), файл
`yellow_tripdata_2024-01.parquet`, [словник даних](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)).

**Вибірка:** 13 колонок, 100 000 рядків — відтворювана випадкова вибірка `sample(n=100_000, random_state=42)` з усього місяця
(перші 100 000 рядків файлу покривають лише 1–3 січня й не містять пропусків, тому не репрезентативні).
""")

# ------------------------------------------------------------------ 1
md("## Завдання 1. Setup і завантаження")
code('''
import glob
import os
import subprocess
import sys


def pip_install(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


pip_install("psycopg2-binary", "sqlalchemy", "pandas", "pyarrow", "fasteners", "platformdirs", "psutil")

# pgserver 0.1.4 має wheel-файли лише до cp312; його розширення зібране в abi3,
# тому на новішому Python (Colab) встановлюємо cp312-wheel з тегом abi3.
if sys.version_info < (3, 13):
    pip_install("pgserver==0.1.4")
else:
    wheel_dir = "/tmp/pgserver_wheel"
    subprocess.check_call([
        sys.executable, "-m", "pip", "download", "-q", "pgserver==0.1.4",
        "--no-deps", "--only-binary=:all:", "--python-version", "3.12", "-d", wheel_dir,
    ])
    for whl in glob.glob(f"{wheel_dir}/pgserver-0.1.4-cp312-cp312-*.whl"):
        os.replace(whl, whl.replace("-cp312-cp312-", "-cp312-abi3-"))
    pip_install("--no-deps", *glob.glob(f"{wheel_dir}/pgserver-0.1.4-cp312-abi3-*.whl"))
''')
code('''
import pandas as pd
import pgserver
from sqlalchemy import create_engine, text

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)

pg = pgserver.get_server("/tmp/hw3_pg", cleanup_mode="stop")
engine = create_engine(pg.get_uri().replace("postgresql://", "postgresql+psycopg2://", 1), future=True)

with engine.connect() as conn:
    version = conn.execute(text("SELECT version()")).scalar_one()
    print(version)


def q(sql_text):
    """Виконати SELECT і повернути результат як DataFrame."""
    return pd.read_sql(text(sql_text), engine)


def run(sql_text):
    """Виконати DDL/DML в одній транзакції."""
    with engine.begin() as conn:
        conn.execute(text(sql_text))
''')
md("""
### Отримання даних з офіційного джерела

Завантажуємо один місячний Parquet-файл (не весь рік), залишаємо 13 потрібних колонок і формуємо контрольований CSV.
Часові колонки в Parquet — це «наївний» локальний час Нью-Йорка без часового поясу; у CSV вони записуються як
`YYYY-MM-DD HH:MM:SS`.
""")
code('''
from urllib.request import urlretrieve

DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
DATA_DIR = "/content" if os.path.isdir("/content") else os.path.abspath("../data")
os.makedirs(DATA_DIR, exist_ok=True)

parquet_path = os.path.join(DATA_DIR, "hw3_taxi_2024_01.parquet")
if not os.path.exists(parquet_path):
    urlretrieve(DATA_URL, parquet_path)

columns = [
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "passenger_count", "trip_distance", "RatecodeID", "store_and_fwd_flag",
    "PULocationID", "DOLocationID", "payment_type",
    "fare_amount", "tip_amount", "total_amount",
]

full = pd.read_parquet(parquet_path, columns=columns)
df = full.sample(n=100_000, random_state=42).sort_values("tpep_pickup_datetime")

csv_path = os.path.join(DATA_DIR, "hw3_taxi_sample.csv")
df.to_csv(csv_path, index=False)

print("рядків у місячному файлі:", len(full))
print("вибірка:", df.shape)
df.head()
''')
md("""
### Staging-таблиця і `COPY FROM STDIN`

Усі сирі колонки мають тип `TEXT`, тому імпорт не впаде через порожні рядки, `1.0` замість цілого числа чи нестандартну дату.
""")
code('''
run("""
DROP TABLE IF EXISTS hw3_taxi_trips CASCADE;
DROP TABLE IF EXISTS hw3_taxi_staging CASCADE;

CREATE TABLE hw3_taxi_staging (
    vendor_id_raw          TEXT,
    pickup_raw             TEXT,
    dropoff_raw            TEXT,
    passenger_count_raw    TEXT,
    trip_distance_raw      TEXT,
    ratecode_id_raw        TEXT,
    store_and_fwd_flag_raw TEXT,
    pu_location_id_raw     TEXT,
    do_location_id_raw     TEXT,
    payment_type_raw       TEXT,
    fare_amount_raw        TEXT,
    tip_amount_raw         TEXT,
    total_amount_raw       TEXT
);
""")

raw_conn = engine.raw_connection()
try:
    with raw_conn.cursor() as cur:
        with open(csv_path, "r", encoding="utf-8", newline="") as file:
            cur.copy_expert(
                """
                COPY hw3_taxi_staging (
                    vendor_id_raw, pickup_raw, dropoff_raw, passenger_count_raw,
                    trip_distance_raw, ratecode_id_raw, store_and_fwd_flag_raw,
                    pu_location_id_raw, do_location_id_raw, payment_type_raw,
                    fare_amount_raw, tip_amount_raw, total_amount_raw
                )
                FROM STDIN
                WITH (FORMAT csv, HEADER true, DELIMITER ',')
                """,
                file,
            )
    raw_conn.commit()
finally:
    raw_conn.close()

q("SELECT COUNT(*) AS loaded_rows FROM hw3_taxi_staging")
''')

# ------------------------------------------------------------------ 2
md("""
## Завдання 2. Staging → типізована таблиця

| Колонка | Семантика | Джерельний формат | Тип | Правила очищення |
|---|---|---|---|---|
| `trip_id` | технічний surrogate key | — | `BIGINT IDENTITY PK` | генерується БД; природного ключа поїздки у файлі немає |
| `vendor_id` | постачальник TPEP-системи (1 = Creative Mobile, 2 = VeriFone) | ціле число як текст | `SMALLINT NOT NULL` | `TRIM` → `NULLIF('')` → `::SMALLINT` |
| `pickup_at`, `dropoff_at` | моменти посадки й висадки | `YYYY-MM-DD HH:MM:SS`, локальний час Нью-Йорка без поясу | `TIMESTAMPTZ NOT NULL` | `::TIMESTAMP AT TIME ZONE 'America/New_York'` — момент прив'язується до поясу NYC (з урахуванням DST) |
| `pickup_date` | календарна дата поїздки в NYC | похідна від `pickup_raw` | `DATE NOT NULL` | `::TIMESTAMP::DATE` — локальна дата без часу для денних агрегатів |
| `passenger_count` | кількість пасажирів (вводить водій) | дробове число `1.0` або порожньо | `SMALLINT`, `CHECK >= 0` | `::NUMERIC::SMALLINT`; порожньо → `NULL` (не підставляємо 1) |
| `trip_distance` | відстань за таксометром, милі | десяткове | `NUMERIC(10,2) NOT NULL`, `CHECK >= 0` | рядки з від'ємною відстанню відкидаються |
| `rate_code_id` | тариф (1 standard, 2 JFK, 3 Newark, 4 Nassau/Westchester, 5 negotiated, 6 group, 99 unknown) | дробове `1.0` або порожньо | `SMALLINT`, `CHECK IN (1..6, 99)` | `::NUMERIC::SMALLINT`; `99` зберігаємо як явний маркер «невідомо» |
| `store_and_fwd` | чи зберігався запис у пам'яті авто без зв'язку з сервером | `Y` / `N` / порожньо | `BOOLEAN` | `Y → TRUE`, `N → FALSE`, інше → `NULL`; справжня двійкова ознака |
| `pu_location_id`, `do_location_id` | TLC Taxi Zone посадки / висадки | ціле | `INTEGER NOT NULL`, `CHECK BETWEEN 1 AND 265` | `::INTEGER`; 264/265 — «Unknown / Outside NYC» за довідником зон |
| `payment_type` | спосіб оплати (0 flex/невідомо, 1 card, 2 cash, 3 no charge, 4 dispute, 5 unknown, 6 voided) | ціле | `SMALLINT NOT NULL`, `CHECK BETWEEN 0 AND 6` | `::SMALLINT` |
| `fare_amount`, `tip_amount`, `total_amount` | тариф за таксометром, чайові, загальна сума, USD | десяткове | `NUMERIC(10,2)` | грошові суми з точністю до центів; від'ємні значення (коригування/повернення) **не** відкидаємо, а аналізуємо в аудиті |

**Правило фільтрації** під час `INSERT INTO ... SELECT`: відкидаються рядки без часу посадки/висадки, з `dropoff_at < pickup_at`,
з від'ємною відстанню, а також рядки, що порушують доменні `CHECK` (невідомий `rate_code_id`, зона поза 1–265, `payment_type` поза 0–6).
""")
code('''
run("""
DROP TABLE IF EXISTS hw3_taxi_trips CASCADE;

CREATE TABLE hw3_taxi_trips (
    trip_id          BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    vendor_id        SMALLINT       NOT NULL,
    pickup_at        TIMESTAMPTZ    NOT NULL,
    dropoff_at       TIMESTAMPTZ    NOT NULL,
    pickup_date      DATE           NOT NULL,
    passenger_count  SMALLINT       CHECK (passenger_count IS NULL OR passenger_count >= 0),
    trip_distance    NUMERIC(10, 2) NOT NULL CHECK (trip_distance >= 0),
    rate_code_id     SMALLINT       CHECK (rate_code_id IS NULL OR rate_code_id IN (1, 2, 3, 4, 5, 6, 99)),
    store_and_fwd    BOOLEAN,
    pu_location_id   INTEGER        NOT NULL CHECK (pu_location_id BETWEEN 1 AND 265),
    do_location_id   INTEGER        NOT NULL CHECK (do_location_id BETWEEN 1 AND 265),
    payment_type     SMALLINT       NOT NULL CHECK (payment_type BETWEEN 0 AND 6),
    fare_amount      NUMERIC(10, 2),
    tip_amount       NUMERIC(10, 2),
    total_amount     NUMERIC(10, 2),

    CHECK (dropoff_at >= pickup_at)
);

WITH parsed AS (
    SELECT
        NULLIF(TRIM(vendor_id_raw), '')::SMALLINT                               AS vendor_id,
        NULLIF(TRIM(pickup_raw), '')::TIMESTAMP AT TIME ZONE 'America/New_York'  AS pickup_at,
        NULLIF(TRIM(dropoff_raw), '')::TIMESTAMP AT TIME ZONE 'America/New_York' AS dropoff_at,
        NULLIF(TRIM(pickup_raw), '')::TIMESTAMP::DATE                           AS pickup_date,
        NULLIF(TRIM(passenger_count_raw), '')::NUMERIC::SMALLINT                AS passenger_count,
        NULLIF(TRIM(trip_distance_raw), '')::NUMERIC(10, 2)                     AS trip_distance,
        NULLIF(TRIM(ratecode_id_raw), '')::NUMERIC::SMALLINT                    AS rate_code_id,
        CASE UPPER(TRIM(store_and_fwd_flag_raw))
            WHEN 'Y' THEN TRUE
            WHEN 'N' THEN FALSE
        END                                                                     AS store_and_fwd,
        NULLIF(TRIM(pu_location_id_raw), '')::INTEGER                           AS pu_location_id,
        NULLIF(TRIM(do_location_id_raw), '')::INTEGER                           AS do_location_id,
        NULLIF(TRIM(payment_type_raw), '')::SMALLINT                            AS payment_type,
        NULLIF(TRIM(fare_amount_raw), '')::NUMERIC(10, 2)                       AS fare_amount,
        NULLIF(TRIM(tip_amount_raw), '')::NUMERIC(10, 2)                        AS tip_amount,
        NULLIF(TRIM(total_amount_raw), '')::NUMERIC(10, 2)                      AS total_amount
    FROM hw3_taxi_staging
)
INSERT INTO hw3_taxi_trips (
    vendor_id, pickup_at, dropoff_at, pickup_date, passenger_count, trip_distance,
    rate_code_id, store_and_fwd, pu_location_id, do_location_id, payment_type,
    fare_amount, tip_amount, total_amount
)
SELECT
    vendor_id, pickup_at, dropoff_at, pickup_date, passenger_count, trip_distance,
    rate_code_id, store_and_fwd, pu_location_id, do_location_id, payment_type,
    fare_amount, tip_amount, total_amount
FROM parsed
WHERE vendor_id IS NOT NULL
  AND pickup_at IS NOT NULL
  AND dropoff_at IS NOT NULL
  AND dropoff_at >= pickup_at
  AND trip_distance >= 0
  AND (passenger_count IS NULL OR passenger_count >= 0)
  AND (rate_code_id IS NULL OR rate_code_id IN (1, 2, 3, 4, 5, 6, 99))
  AND pu_location_id BETWEEN 1 AND 265
  AND do_location_id BETWEEN 1 AND 265
  AND payment_type BETWEEN 0 AND 6;
""")

q("SELECT * FROM hw3_taxi_trips ORDER BY trip_id LIMIT 5")
''')

# ------------------------------------------------------------------ 3
md("## Завдання 3. Data-quality audit\n\n### 3.1. Контроль кількості рядків")
query("""
SELECT 'staging' AS layer, COUNT(*) AS n_rows FROM hw3_taxi_staging
UNION ALL
SELECT 'clean' AS layer, COUNT(*) AS n_rows FROM hw3_taxi_trips
""")
md("Причини відсіювання (рахуються на сирому шарі за тими самими правилами):")
query("""
SELECT
    COUNT(*) FILTER (WHERE dropoff_raw::TIMESTAMP < pickup_raw::TIMESTAMP)       AS dropoff_before_pickup,
    COUNT(*) FILTER (WHERE trip_distance_raw::NUMERIC < 0)                        AS negative_distance,
    COUNT(*) FILTER (WHERE pu_location_id_raw::INTEGER NOT BETWEEN 1 AND 265
                        OR do_location_id_raw::INTEGER NOT BETWEEN 1 AND 265)     AS bad_location,
    COUNT(*) FILTER (WHERE payment_type_raw::INTEGER NOT BETWEEN 0 AND 6)         AS bad_payment_type,
    COUNT(*) FILTER (WHERE NULLIF(TRIM(ratecode_id_raw), '')::NUMERIC
                           NOT IN (1, 2, 3, 4, 5, 6, 99))                         AS bad_rate_code
FROM hw3_taxi_staging
""")
md(INTERP["3.1"])

md("""
### 3.2. Перевірка дублікатів

Гарантованого природного ключа поїздки у файлі немає. Як business key беремо комбінацію
`(vendor_id, pickup_at, dropoff_at, pu_location_id, do_location_id, total_amount)`: дві різні поїздки одного
постачальника з однаковими секундами посадки й висадки, зонами та сумою практично неможливі. Тож перевірка виявляє
**точні або потенційні** дублікати, а не гарантовані.
""")
query("""
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT (vendor_id, pickup_at, dropoff_at, pu_location_id, do_location_id, total_amount))
        AS distinct_business_keys,
    COUNT(*) - COUNT(DISTINCT (vendor_id, pickup_at, dropoff_at, pu_location_id, do_location_id, total_amount))
        AS duplicate_rows
FROM hw3_taxi_trips
""")
md(INTERP["3.2"])

md("### 3.3. Null-rate і спеціальні маркери\n\nNull-rate у clean-шарі для семи колонок:")
query("""
SELECT
    ROUND(COUNT(*) FILTER (WHERE passenger_count IS NULL) * 100.0 / NULLIF(COUNT(*), 0), 2) AS pct_null_passenger_count,
    ROUND(COUNT(*) FILTER (WHERE rate_code_id    IS NULL) * 100.0 / NULLIF(COUNT(*), 0), 2) AS pct_null_rate_code_id,
    ROUND(COUNT(*) FILTER (WHERE store_and_fwd   IS NULL) * 100.0 / NULLIF(COUNT(*), 0), 2) AS pct_null_store_and_fwd,
    ROUND(COUNT(*) FILTER (WHERE trip_distance   IS NULL) * 100.0 / NULLIF(COUNT(*), 0), 2) AS pct_null_trip_distance,
    ROUND(COUNT(*) FILTER (WHERE fare_amount     IS NULL) * 100.0 / NULLIF(COUNT(*), 0), 2) AS pct_null_fare_amount,
    ROUND(COUNT(*) FILTER (WHERE tip_amount      IS NULL) * 100.0 / NULLIF(COUNT(*), 0), 2) AS pct_null_tip_amount,
    ROUND(COUNT(*) FILTER (WHERE total_amount    IS NULL) * 100.0 / NULLIF(COUNT(*), 0), 2) AS pct_null_total_amount,
    COUNT(*) AS total_rows
FROM hw3_taxi_trips
""")
md("Спеціальні маркери в сирому шарі: порожні значення (COPY перетворює порожнє поле CSV на NULL, тому рахуємо і NULL, і `''`), коди «невідомо» (`payment_type = 0/5`, `RatecodeID = 99`), зони 264/265 (Unknown/Outside NYC):")
query("""
SELECT
    COUNT(*) FILTER (WHERE COALESCE(TRIM(passenger_count_raw), '') = '')    AS empty_passenger_count,
    COUNT(*) FILTER (WHERE COALESCE(TRIM(ratecode_id_raw), '') = '')        AS empty_ratecode,
    COUNT(*) FILTER (WHERE COALESCE(TRIM(store_and_fwd_flag_raw), '') = '') AS empty_store_and_fwd,
    COUNT(*) FILTER (WHERE payment_type_raw IN ('0', '5'))     AS payment_unknown_0_or_5,
    COUNT(*) FILTER (WHERE ratecode_id_raw = '99.0')           AS ratecode_99_unknown,
    COUNT(*) FILTER (WHERE pu_location_id_raw IN ('264', '265')
                        OR do_location_id_raw IN ('264', '265')) AS unknown_zone_264_265,
    COUNT(*) FILTER (WHERE passenger_count_raw = '0.0')        AS zero_passengers,
    COUNT(*)                                                   AS total_rows
FROM hw3_taxi_staging
""")
md(INTERP["3.3"])

# ------------------------------------------------------------------ 4
md("## Завдання 4. Розподіли та потенційні аномалії\n\n### 4.1. Розподіл способів оплати")
query("""
SELECT
    payment_type,
    CASE payment_type
        WHEN 0 THEN 'flex fare / unknown'
        WHEN 1 THEN 'credit card'
        WHEN 2 THEN 'cash'
        WHEN 3 THEN 'no charge'
        WHEN 4 THEN 'dispute'
        WHEN 5 THEN 'unknown'
        WHEN 6 THEN 'voided trip'
    END AS payment_label,
    COUNT(*) AS n_rows,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_rows
FROM hw3_taxi_trips
GROUP BY payment_type
ORDER BY n_rows DESC, payment_type
LIMIT 10
""")
md(INTERP["4.1"])
md("### 4.2. Базова статистика: `trip_distance` і `total_amount`")
query("""
SELECT 'trip_distance' AS metric,
       COUNT(trip_distance) AS n_non_null, MIN(trip_distance) AS min_value, MAX(trip_distance) AS max_value,
       ROUND(AVG(trip_distance), 2) AS avg_value, ROUND(STDDEV_SAMP(trip_distance), 2) AS sd_value
FROM hw3_taxi_trips
UNION ALL
SELECT 'total_amount',
       COUNT(total_amount), MIN(total_amount), MAX(total_amount),
       ROUND(AVG(total_amount), 2), ROUND(STDDEV_SAMP(total_amount), 2)
FROM hw3_taxi_trips
""")
md(INTERP["4.2"])
md("""
### 4.3. Domain outlier-check

Правила, що суперечать предметній області: від'ємні суми, дата посадки поза січнем 2024, поїздка довша за 12 годин,
відстань понад 100 миль, ненульова оплата при нульовій відстані й тривалості менше хвилини.
""")
query("""
SELECT
    COUNT(*) FILTER (WHERE fare_amount < 0 OR tip_amount < 0 OR total_amount < 0)             AS negative_amounts,
    COUNT(*) FILTER (WHERE pickup_date NOT BETWEEN DATE '2024-01-01' AND DATE '2024-01-31')    AS pickup_outside_month,
    COUNT(*) FILTER (WHERE dropoff_at - pickup_at > INTERVAL '12 hours')                       AS longer_than_12h,
    COUNT(*) FILTER (WHERE trip_distance > 100)                                                AS distance_over_100mi,
    COUNT(*) FILTER (WHERE trip_distance = 0 AND dropoff_at - pickup_at < INTERVAL '1 minute'
                       AND total_amount > 0)                                                   AS zero_trip_but_charged
FROM hw3_taxi_trips
""")
query("""
SELECT trip_id, pickup_at, dropoff_at, trip_distance, payment_type, fare_amount, tip_amount, total_amount
FROM hw3_taxi_trips
WHERE fare_amount < 0
   OR total_amount < 0
   OR pickup_date NOT BETWEEN DATE '2024-01-01' AND DATE '2024-01-31'
   OR trip_distance > 100
ORDER BY pickup_at
LIMIT 20
""")
md(INTERP["4.3"])

# ------------------------------------------------------------------ 5
md("## Завдання 5. Десять DQL/EDA-запитів")
QUERIES = [
    ("Q1", "У які години доби (за часом Нью-Йорка) починається найбільше поїздок?", """
SELECT
    EXTRACT(HOUR FROM pickup_at AT TIME ZONE 'America/New_York')::INT AS pickup_hour,
    COUNT(*) AS n_trips
FROM hw3_taxi_trips
GROUP BY pickup_hour
ORDER BY n_trips DESC, pickup_hour
LIMIT 5
"""),
    ("Q2", "Яка частка поїздок кожного постачальника має невідому кількість пасажирів (NULL passenger_count)?", """
SELECT
    vendor_id,
    COUNT(*) AS n_trips,
    COUNT(*) FILTER (WHERE passenger_count IS NULL) AS n_null_passengers,
    ROUND(COUNT(*) FILTER (WHERE passenger_count IS NULL) * 100.0 / COUNT(*), 2) AS pct_null_passengers
FROM hw3_taxi_trips
GROUP BY vendor_id
ORDER BY vendor_id
"""),
    ("Q3", "Які поїздки мають від'ємні суми або нульову відстань, хоча це не безкоштовна й не спірна поїздка?", """
SELECT
    trip_id, pickup_at, trip_distance, payment_type, fare_amount, total_amount
FROM hw3_taxi_trips
WHERE (fare_amount < 0 OR total_amount < 0)
   OR (trip_distance = 0 AND NOT (payment_type IN (3, 4)) AND total_amount > 0)
ORDER BY total_amount
LIMIT 10
"""),
    ("Q4", "Чим відрізняються середній тариф і чайові для поїздок з аеропортів JFK (132) і LaGuardia (138) від решти?", """
SELECT
    CASE WHEN pu_location_id IN (132, 138) THEN 'airport pickup' ELSE 'other' END AS pickup_group,
    COUNT(*) AS n_trips,
    ROUND(AVG(trip_distance), 2) AS avg_distance,
    ROUND(AVG(fare_amount), 2) AS avg_fare,
    ROUND(AVG(tip_amount), 2) AS avg_tip
FROM hw3_taxi_trips
GROUP BY pickup_group
ORDER BY n_trips DESC
"""),
    ("Q5", "Яку частку становлять довгі поїздки на 10–30 миль і скільки вони в середньому коштують?", """
SELECT
    COUNT(*) FILTER (WHERE trip_distance BETWEEN 10 AND 30) AS n_long_trips,
    ROUND(COUNT(*) FILTER (WHERE trip_distance BETWEEN 10 AND 30) * 100.0 / COUNT(*), 2) AS pct_long_trips,
    ROUND(AVG(total_amount) FILTER (WHERE trip_distance BETWEEN 10 AND 30), 2) AS avg_total_long,
    ROUND(AVG(total_amount) FILTER (WHERE trip_distance < 10), 2) AS avg_total_short
FROM hw3_taxi_trips
"""),
    ("Q6", "Скільки поїздок у файлі за січень 2024 насправді датовані іншим місяцем?", """
SELECT
    pickup_date,
    COUNT(*) AS n_trips
FROM hw3_taxi_trips
WHERE pickup_date::TEXT NOT LIKE '2024-01-%'
GROUP BY pickup_date
ORDER BY pickup_date
"""),
    ("Q7", "Який середній відсоток чайових від тарифу при оплаті карткою та готівкою?", """
SELECT
    payment_type,
    COUNT(*) AS n_trips,
    ROUND(AVG(tip_amount / NULLIF(fare_amount, 0)) * 100, 2) AS avg_tip_pct,
    ROUND(AVG(COALESCE(passenger_count, 1)), 2) AS avg_passengers_null_as_1
FROM hw3_taxi_trips
WHERE payment_type IN (1, 2)
  AND fare_amount > 0
GROUP BY payment_type
ORDER BY payment_type
"""),
    ("Q8", "Яка частка поїздок починається й закінчується в тій самій зоні і яка в них середня відстань?", """
SELECT
    pu_location_id IS NOT DISTINCT FROM do_location_id AS same_zone,
    COUNT(*) AS n_trips,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_trips,
    ROUND(AVG(trip_distance), 2) AS avg_distance
FROM hw3_taxi_trips
GROUP BY same_zone
ORDER BY same_zone
"""),
    ("Q9", "Які комбінації тарифу й способу оплати трапляються в поїздках із невідомою кількістю пасажирів?", """
SELECT DISTINCT
    rate_code_id,
    payment_type,
    store_and_fwd
FROM hw3_taxi_trips
WHERE passenger_count IS NULL
ORDER BY rate_code_id NULLS FIRST, payment_type
"""),
    ("Q10", "Які маршрути «зона посадки → зона висадки» найпопулярніші, якщо виключити невідомі зони 264/265?", """
SELECT
    pu_location_id,
    do_location_id,
    COUNT(*) AS n_trips,
    ROUND(AVG(total_amount), 2) AS avg_total
FROM hw3_taxi_trips
WHERE NOT (pu_location_id IN (264, 265) OR do_location_id IN (264, 265))
  AND pu_location_id IS DISTINCT FROM do_location_id
GROUP BY pu_location_id, do_location_id
ORDER BY n_trips DESC, pu_location_id, do_location_id
LIMIT 10
"""),
]
for qid, question, sql_text in QUERIES:
    md(f"### {qid}\n\n**Business-question:** {question}")
    query(sql_text)
    md(f"**Interpretation:** {INTERP[qid]}")

md("""
### Покриття обов'язкових конструкцій

| Конструкція | Запити |
|---|---|
| явний перелік колонок у `SELECT` | Q3, Q6, Q9, Q10 (і всі інші) |
| `WHERE` з `AND` / `OR` / `NOT` і дужками | Q3, Q10 |
| `IN` / `BETWEEN` | Q3, Q4, Q7, Q10 / Q5 |
| `LIKE` | Q6 |
| `IS NULL` / `IS NOT NULL` | Q2, Q9 |
| `COALESCE` / `NULLIF` | Q7 |
| `IS DISTINCT FROM` / `IS NOT DISTINCT FROM` | Q10 / Q8 |
| `DISTINCT` | Q9 |
| `ORDER BY` + `LIMIT` | Q1, Q3, Q10 |
| `GROUP BY` (≥ 2) | Q1, Q2, Q4, Q6, Q7, Q8, Q10 |
""")

md("## Завдання 6. Reflection\n\n" + INTERP["reflection"])

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"}, "colab": {"provenance": []}}
nbf.write(nb, ROOT / "notebooks/hw3_nyc_taxi.ipynb")
print("wrote notebook")
