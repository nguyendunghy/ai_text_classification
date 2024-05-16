from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.sql.database import SessionLocal
from src.sql.models import TextModel


def query_all_data():
    db = SessionLocal()
    rows = db.query(TextModel).all()
    db.close()
    rows = [row.__dict__ for row in rows]
    return rows


def save_to_csv(rows, output_csv):
    df = pd.DataFrame.from_records(rows)
    df = df.drop(['_sa_instance_state'], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(str(output_csv))
    print(f"Saved {df.shape} rows to {output_csv}")


def parse():
    parser = ArgumentParser()
    parser.add_argument("--output_csv", type=Path, default="resources/human_data.pkl")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    rows = query_all_data()
    save_to_csv(rows, args.output_csv)
