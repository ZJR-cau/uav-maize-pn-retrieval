"""Isolate test samples to prevent leakage into model development."""
import pandas as pd
from pathlib import Path

FEATURE_FILE = Path(r"G:\PhyMaize\features\features-Pn.xlsx")
SPLIT_ROOT = Path(r"G:\PhyMaize\features\splits")
OUTPUT_ROOT = FEATURE_FILE.parent

SHEET_NAME = "features"
ID_COL = "Sample_ID"

TRAIN_FILE = SPLIT_ROOT / "train.csv"
VAL_FILE = SPLIT_ROOT / "val.csv"
TEST_FILE = SPLIT_ROOT / "test.csv"

TRAIN_VAL_OUTPUT = OUTPUT_ROOT / "features-Pn (train and val).xlsx"
TEST_OUTPUT = OUTPUT_ROOT / "features-Pn (test).xlsx"


def main():
    """Run the main analysis workflow."""
    df = pd.read_excel(FEATURE_FILE, sheet_name=SHEET_NAME)
    train_ids = pd.read_csv(TRAIN_FILE)[ID_COL].astype(str).str.strip()
    val_ids = pd.read_csv(VAL_FILE)[ID_COL].astype(str).str.strip()
    test_ids = pd.read_csv(TEST_FILE)[ID_COL].astype(str).str.strip()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    train_val_ids = set(train_ids) | set(val_ids)
    test_ids = set(test_ids)
    train_val_df = df[df[ID_COL].isin(train_val_ids)].copy()
    test_df = df[df[ID_COL].isin(test_ids)].copy()
    train_val_df.to_excel(TRAIN_VAL_OUTPUT, index=False)
    test_df.to_excel(TEST_OUTPUT, index=False)
    print("=" * 60)
    print("数据集划分完成")
    print("=" * 60)
    print(f"原始数据集 : {len(df)}")
    print(f"Train      : {len(train_ids)}")
    print(f"Validation : {len(val_ids)}")
    print(f"Test       : {len(test_ids)}")
    print(f"Train+Val  : {len(train_val_df)}")
    print(f"Test       : {len(test_df)}")
    print(f"\nTrain+Val 文件：{TRAIN_VAL_OUTPUT}")
    print(f"Test 文件     ：{TEST_OUTPUT}")
    if len(train_val_df) != len(train_val_ids):
        print(f"\n警告：Train+Val 中有 {len(train_val_ids) - len(train_val_df)} 个 Sample_ID 未在特征表中找到。")
    if len(test_df) != len(test_ids):
        print(f"警告：Test 中有 {len(test_ids) - len(test_df)} 个 Sample_ID 未在特征表中找到。")
    if train_val_ids & test_ids:
        print("警告：Train+Val 与 Test 存在重复 Sample_ID，请检查数据划分。")


if __name__ == "__main__":
    main()