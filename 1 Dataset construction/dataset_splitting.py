"""Split the prepared dataset into training, validation, and test subsets."""
import os
import random
import pandas as pd

# ============================================================

# ============================================================

dataset_root = r"G:\PhyMaize"

data_dir = os.path.join(dataset_root, "image data")
splits_dir = os.path.join(dataset_root, "features", "splits")


os.makedirs(splits_dir, exist_ok=True)

# ============================================================

# ============================================================

sample_ids = []

for name in os.listdir(data_dir):

    sample_path = os.path.join(data_dir, name)


    if os.path.isdir(sample_path):
        sample_ids.append(name)

# ============================================================

# ============================================================

sample_ids = sorted(sample_ids)

print(f"检测到样本总数：{len(sample_ids)}")

# ============================================================

# ============================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


random.shuffle(sample_ids)

# ============================================================


#    Train      = 70%
#    Validation = 15%
#    Test       = 15%
# ============================================================

total_num = len(sample_ids)

train_num = int(total_num * 0.80)
validation_num = int(total_num * 0.1)


test_num = total_num - train_num - validation_num

# ============================================================

# ============================================================

train_ids = sample_ids[:train_num]

validation_ids = sample_ids[
    train_num: train_num + validation_num
]

test_ids = sample_ids[
    train_num + validation_num:
]

# ============================================================

# ============================================================

train_df = pd.DataFrame({
    "Sample_ID": train_ids
})

validation_df = pd.DataFrame({
    "Sample_ID": validation_ids
})

test_df = pd.DataFrame({
    "Sample_ID": test_ids
})

train_path = os.path.join(
    splits_dir,
    "train.csv"
)

validation_path = os.path.join(
    splits_dir,
    "val.csv"
)

test_path = os.path.join(
    splits_dir,
    "test.csv"
)

train_df.to_csv(
    train_path,
    index=False
)

validation_df.to_csv(
    validation_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)

# ============================================================

# ============================================================

print("\n数据集划分完成！")

print(f"\n总样本数：{total_num}")

print(
    f"训练集：{len(train_ids)} "
    f"({len(train_ids) / total_num * 100:.2f}%)"
)

print(
    f"验证集：{len(validation_ids)} "
    f"({len(validation_ids) / total_num * 100:.2f}%)"
)

print(
    f"测试集：{len(test_ids)} "
    f"({len(test_ids) / total_num * 100:.2f}%)"
)

print("\n保存路径：")

print(train_path)
print(validation_path)
print(test_path)