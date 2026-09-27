# goit-rdb-hw-03 — NYC Yellow Taxi: завантаження, аудит якості та EDA у PostgreSQL

**Автор:** Samoilenko Mariia

## Датасет

- **Назва:** NYC TLC Trip Record Data — Yellow Taxi, січень 2024
- **Офіційне джерело:** https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Файл:** https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet (2 964 624 рядки)
- **Словник даних:** https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf

## Формування вибірки

1. Notebook завантажує місячний Parquet-файл з офіційного джерела (весь рік не завантажується).
2. Залишаються 13 колонок: `VendorID`, `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `passenger_count`, `trip_distance`, `RatecodeID`, `store_and_fwd_flag`, `PULocationID`, `DOLocationID`, `payment_type`, `fare_amount`, `tip_amount`, `total_amount`.
3. Береться відтворювана випадкова вибірка `sample(n=100_000, random_state=42)` з усього місяця й зберігається в CSV `data/hw3_taxi_sample.csv`.
4. CSV завантажується в PostgreSQL через `COPY FROM STDIN` у staging-таблицю `hw3_taxi_staging` (100 000 рядків), далі очищується в типізовану `hw3_taxi_trips` (99 999 рядків).

## Структура

| Шлях | Опис |
|---|---|
| `notebooks/hw3_nyc_taxi.ipynb` | Notebook з outputs: setup, завантаження, DDL, аудит, розподіли, 10 DQL/EDA-запитів, reflection |
| `data/hw3_taxi_sample.csv` | Контрольований CSV (100 000 рядків, ~8 МБ) |
| `tools/build_notebook.py`, `tools/interpretations.json` | Збирання notebook |

## Запуск

Відкрийте notebook у Google Colab і виконайте **Runtime → Restart session and run all**.
PostgreSQL 16 запускається всередині notebook через `pgserver`; дані завантажуються автоматично з офіційного URL.
