import pandas as pd
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TXT_PATH = "corrections/flipped_1501-3000.txt"
CSV_IN_PATH = "data/raw/train.csv"
CSV_OUT_PATH = "data/processed/train.csv"
IMG_DIR = "data/raw/faces-spring-2020/faces-spring-2020"


with open(TXT_PATH, 'r') as f:
    flipped_ids = set(int(line.strip()) for line in f)

df = pd.read_csv(CSV_IN_PATH)

flip_mask = df["id"].isin(flipped_ids)
df.loc[flip_mask, "glasses"] = 1 - df.loc[flip_mask, "glasses"]

df.to_csv(CSV_OUT_PATH, index=False)
print(f"Corrected {flip_mask.sum()} entries.")