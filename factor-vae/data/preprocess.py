from zipfile import ZipFile

import polars as pl
import pandas as pd
from data.utils import DATA_DIR


def extract_zip(locale: str):
    for zip in DATA_DIR.iterdir():
        if zip.suffix == ".zip" and locale in zip.stem:
            with ZipFile(zip, "r") as zip_file:
                zip_file.extractall(DATA_DIR)
            CURRENT_DIR = DATA_DIR / "data" / "daily" / locale
            DEST_DIR = DATA_DIR / locale
            DEST_DIR.mkdir(exist_ok=True)
            for dir in CURRENT_DIR.iterdir():
                for file in dir.iterdir():
                    if file.is_dir():
                        for f in file.iterdir():
                            if f.name.split(".")[-2] == locale:
                                _ = f.replace(f"{DEST_DIR}/{f.name}")
                    else:
                        if file.name.split(".")[-2] == locale:
                            _ = file.replace(f"{DEST_DIR}/{file.name}")
    print(f"Extracted {locale}.zip")


def make_csv(locale: str):
    locale_dir = DATA_DIR / locale
    tmp = pl.DataFrame()
    column_name = ['<TICKER>', '<PER>', '<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>', '<OPENINT>']
    column_type = [pl.String, pl.String, pl.Int64, pl.Int64, pl.Float64, pl.Float64, pl.Float64, pl.Float64, pl.Float64, pl.Int64]
    dtypes = {}
    for idx in range(len(column_name)):
        dtypes[column_name[idx]] = column_type[idx]
    for idx, file in enumerate(locale_dir.iterdir()):
        try:
            data = pl.read_csv(file, dtypes = dtypes)
            tmp = pl.concat([tmp, data])
            print(idx)
        except pl.NoDataError:
            pass

    tmp.write_csv(DATA_DIR / f"{locale}.csv")
    print(f"Saved {locale}.csv")

def main(locale: str):
    if not (DATA_DIR / locale).exists():
        extract_zip(locale)
    make_csv(locale)


if __name__ == "__main__":
    for locale in ["hk"]:  #, "hu", "jp", "pl", "uk", "us"
        main(locale)
