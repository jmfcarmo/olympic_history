import hashlib
import sqlite3
from datetime import date
from typing import Optional

import pandas as pd


TODAY = date.today().isoformat()


# --- BRONZE: load raw CSVs as-is ---

def bronze_events() -> pd.DataFrame:
    return pd.read_csv("athlete_events.csv")


def bronze_regions() -> pd.DataFrame:
    return pd.read_csv("noc_regions.csv")


# --- SILVER: clean and prepare ---

def silver_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(subset=["Name", "Event", "Year", "Season", "NOC"])
    df["Year"] = df["Year"].astype(int)
    df["Medal"] = df["Medal"].fillna("None")
    return df


def silver_regions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["NOC"])
    df = df.rename(columns={"region": "country"})
    return df


# --- GOLD: build star schema ---

def create_schema(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS dim_date (
            date_id INTEGER PRIMARY KEY AUTOINCREMENT,
            year    INTEGER NOT NULL,
            season  TEXT NOT NULL,
            UNIQUE(year, season)
        );
        CREATE TABLE IF NOT EXISTS dim_region (
            region_id INTEGER PRIMARY KEY AUTOINCREMENT,
            noc       TEXT NOT NULL UNIQUE,
            country   TEXT,
            notes     TEXT
        );
        CREATE TABLE IF NOT EXISTS dim_event (
            event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name TEXT NOT NULL,
            year       INTEGER NOT NULL,
            city       TEXT,
            season     TEXT,
            sport      TEXT,
            UNIQUE(event_name, year, season)
        );
        CREATE TABLE IF NOT EXISTS dim_athlete (
            surrogate_key  INTEGER PRIMARY KEY AUTOINCREMENT,
            natural_key    TEXT NOT NULL,
            name           TEXT NOT NULL,
            sex            TEXT,
            age            REAL,
            height         REAL,
            weight         REAL,
            row_hash       TEXT NOT NULL,
            effective_from TEXT NOT NULL,
            effective_to   TEXT NOT NULL DEFAULT '9999-12-31',
            is_current     INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS fact_olympic_results (
            result_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id INTEGER NOT NULL REFERENCES dim_athlete(surrogate_key),
            event_id   INTEGER NOT NULL REFERENCES dim_event(event_id),
            region_id  INTEGER NOT NULL REFERENCES dim_region(region_id),
            date_id    INTEGER NOT NULL REFERENCES dim_date(date_id),
            medal      INTEGER NOT NULL DEFAULT 0
        );
    """)


def load_dim_date(db: sqlite3.Connection, df: pd.DataFrame) -> dict[tuple[int, str], int]:
    # SCD1 - year and season are immutable
    mapping: dict[tuple[int, str], int] = {}
    cur = db.cursor()
    for year, season in df[["Year", "Season"]].drop_duplicates().itertuples(index=False):
        cur.execute("INSERT OR IGNORE INTO dim_date (year, season) VALUES (?,?)", (int(year), str(season)))
        cur.execute("SELECT date_id FROM dim_date WHERE year=? AND season=?", (int(year), str(season)))
        row = cur.fetchone()
        if row:
            mapping[(int(year), str(season))] = int(row[0])
    db.commit()
    return mapping


def load_dim_region(db: sqlite3.Connection, df: pd.DataFrame) -> dict[str, int]:
    # SCD1 - overwrite country and notes on change
    mapping: dict[str, int] = {}
    cur = db.cursor()
    for _, row in df.iterrows():
        noc = str(row["NOC"])
        country: Optional[str] = row.get("country") or None
        notes: Optional[str] = row.get("notes") or None
        cur.execute("SELECT region_id FROM dim_region WHERE noc=?", (noc,))
        existing = cur.fetchone()
        if existing:
            cur.execute("UPDATE dim_region SET country=?, notes=? WHERE noc=?", (country, notes, noc))
            mapping[noc] = int(existing[0])
        else:
            cur.execute("INSERT INTO dim_region (noc, country, notes) VALUES (?,?,?)", (noc, country, notes))
            mapping[noc] = int(cur.lastrowid)  # type: ignore[arg-type]
    db.commit()
    return mapping


def load_dim_event(db: sqlite3.Connection, df: pd.DataFrame) -> dict[tuple[str, int, str], int]:
    # SCD1 - overwrite city and sport on change
    mapping: dict[tuple[str, int, str], int] = {}
    cur = db.cursor()
    for row in df[["Event", "Year", "City", "Season", "Sport"]].drop_duplicates().itertuples(index=False):
        key = (str(row.Event), int(row.Year), str(row.Season))
        cur.execute("SELECT event_id FROM dim_event WHERE event_name=? AND year=? AND season=?", key)
        existing = cur.fetchone()
        if existing:
            cur.execute("UPDATE dim_event SET city=?, sport=? WHERE event_id=?", (str(row.City), str(row.Sport), existing[0]))
            mapping[key] = int(existing[0])
        else:
            cur.execute(
                "INSERT INTO dim_event (event_name, year, city, season, sport) VALUES (?,?,?,?,?)",
                (str(row.Event), int(row.Year), str(row.City), str(row.Season), str(row.Sport)),
            )
            mapping[key] = int(cur.lastrowid)  # type: ignore[arg-type]
    db.commit()
    return mapping


def load_dim_athlete(db: sqlite3.Connection, df: pd.DataFrame) -> dict[str, int]:
    # SCD2 on age, height, weight - new row created when these change
    # SCD1 on name, sex - overwritten in place
    mapping: dict[str, int] = {}
    cur = db.cursor()
    for row in df[["Name", "Sex", "Age", "Height", "Weight"]].drop_duplicates(subset=["Name"]).itertuples(index=False):
        natural_key = str(row.Name)
        name = str(row.Name)
        sex = str(row.Sex) if pd.notna(row.Sex) else ""
        age: Optional[float] = float(row.Age) if pd.notna(row.Age) else None
        height: Optional[float] = float(row.Height) if pd.notna(row.Height) else None
        weight: Optional[float] = float(row.Weight) if pd.notna(row.Weight) else None
        new_hash = hashlib.md5(f"{age}|{height}|{weight}".encode()).hexdigest()

        cur.execute("SELECT surrogate_key, row_hash, name, sex FROM dim_athlete WHERE natural_key=? AND is_current=1", (natural_key,))
        existing = cur.fetchone()

        if not existing:
            cur.execute(
                "INSERT INTO dim_athlete (natural_key, name, sex, age, height, weight, row_hash, effective_from) VALUES (?,?,?,?,?,?,?,?)",
                (natural_key, name, sex, age, height, weight, new_hash, TODAY),
            )
            mapping[natural_key] = int(cur.lastrowid)  # type: ignore[arg-type]
        else:
            surr_key, old_hash, old_name, old_sex = existing
            if old_hash != new_hash:
                cur.execute("UPDATE dim_athlete SET effective_to=?, is_current=0 WHERE surrogate_key=?", (TODAY, surr_key))
                cur.execute(
                    "INSERT INTO dim_athlete (natural_key, name, sex, age, height, weight, row_hash, effective_from) VALUES (?,?,?,?,?,?,?,?)",
                    (natural_key, name, sex, age, height, weight, new_hash, TODAY),
                )
                mapping[natural_key] = int(cur.lastrowid)  # type: ignore[arg-type]
            elif old_name != name or old_sex != sex:
                cur.execute("UPDATE dim_athlete SET name=?, sex=? WHERE surrogate_key=?", (name, sex, surr_key))
                mapping[natural_key] = int(surr_key)
            else:
                mapping[natural_key] = int(surr_key)
    db.commit()
    return mapping


def load_fact(
    db: sqlite3.Connection,
    df: pd.DataFrame,
    athlete_map: dict[str, int],
    event_map: dict[tuple[str, int, str], int],
    region_map: dict[str, int],
    date_map: dict[tuple[int, str], int],
) -> None:
    cur = db.cursor()
    medals = {"Gold", "Silver", "Bronze"}
    for row in df.itertuples(index=False):
        athlete_id = athlete_map.get(str(row.Name))
        event_id = event_map.get((str(row.Event), int(row.Year), str(row.Season)))
        region_id = region_map.get(str(row.NOC))
        date_id = date_map.get((int(row.Year), str(row.Season)))
        if athlete_id and event_id and region_id and date_id:
            cur.execute(
                "INSERT INTO fact_olympic_results (athlete_id, event_id, region_id, date_id, medal) VALUES (?,?,?,?,?)",
                (athlete_id, event_id, region_id, date_id, 1 if str(row.Medal) in medals else 0),
            )
    db.commit()


# --- PIPELINE ---

def run_pipeline() -> None:
    print("Bronze: loading raw data...")
    raw_events = bronze_events()
    raw_regions = bronze_regions()

    print("Silver: cleaning data...")
    clean_events = silver_events(raw_events)
    clean_regions = silver_regions(raw_regions)

    print("Gold: building star schema...")
    db = sqlite3.connect("olympic_warehouse.db")
    db.execute("PRAGMA foreign_keys=ON;")
    create_schema(db)

    date_map    = load_dim_date(db, clean_events)
    region_map  = load_dim_region(db, clean_regions)
    event_map   = load_dim_event(db, clean_events)
    athlete_map = load_dim_athlete(db, clean_events)
    load_fact(db, clean_events, athlete_map, event_map, region_map, date_map)

    db.close()
    print("Done! Warehouse saved to olympic_warehouse.db")


if __name__ == "__main__":
    run_pipeline()
