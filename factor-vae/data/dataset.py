import os
from typing import Tuple
import numpy as np
import polars as pl
import torch
from torch.utils import data as dt

class StockReturnDataset(dt.Dataset):
    def __init__(self, locale: str, len_hist: int, num_stocks: int):
        super().__init__()
        self.len_hist = len_hist

        # Set DATA_DIR to match the one in your data generation script
        DATA_DIR = './data'  # Ensure this matches the DATA_DIR in your data generation script

        # Construct the full path to the CSV file
        locale_path = os.path.join(DATA_DIR, f"{locale}.csv")

        # Check if the file exists
        if not os.path.exists(locale_path):
            raise FileNotFoundError(f"Data file {locale_path} not found.")
            # Alternatively, you can call a method to generate the data here

        # Read and process the data
        df = (
            pl.read_csv(locale_path)
            .select(["<DATE>", "<TICKER>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"])
            .with_columns([
                pl.col("<CLOSE>").pct_change().alias("<RETURN>"),
                pl.col("<OPEN>").cast(pl.Float64),
                pl.col("<HIGH>").cast(pl.Float64),
                pl.col("<LOW>").cast(pl.Float64),
                pl.col("<CLOSE>").cast(pl.Float64),
                pl.col("<VOL>").cast(pl.Float64),
            ])
            .drop_nulls()
            .sort(["<DATE>", "<TICKER>"])
        )

        # Prepare the data for the model
        val_cols = ["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>", "<RETURN>"]
        data_list = []
        for val_col in val_cols:
            pivoted = (
                df.pivot(index="<DATE>", columns="<TICKER>", values=val_col)
                .fill_null(0)
                .sort("<DATE>")
                .select(pl.exclude("<DATE>"))
            )
            data_array = pivoted.to_numpy().astype(np.float32)
            data_list.append(data_array)

        # Stack the data arrays along the third axis
        self.data = np.stack(data_list, axis=2)  # Shape: [num_dates, num_tickers, num_features]

        # Adjust num_stocks if necessary
        self.num_stocks = min(num_stocks, self.data.shape[1])

        # For debugging: print data shapes
        print(f"Data shape: {self.data.shape}")
        print(f"Number of dates (time steps): {self.data.shape[0]}")
        print(f"Number of tickers (stocks): {self.data.shape[1]}")
        print(f"Number of features: {self.data.shape[2]}")
        print(f"Using num_stocks: {self.num_stocks}")
        print(f"Calculated dataset length: {len(self)}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Corrected index check
        if idx + self.len_hist >= self.data.shape[0]:
            raise IndexError("Index out of range")

        input_data = self.data[idx : idx + self.len_hist, : self.num_stocks, :]
        target_data = self.data[idx + self.len_hist, : self.num_stocks, -1]  # '<RETURN>' column

        # Convert to torch.Tensor
        input_tensor = torch.from_numpy(input_data)
        target_tensor = torch.from_numpy(target_data)

        return input_tensor, target_tensor

    def __len__(self) -> int:
        return self.data.shape[0] - self.len_hist
