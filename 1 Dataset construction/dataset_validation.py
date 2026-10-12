"""Validate dataset integrity, feature files, labels, and data partitions."""
# ================================================================
# dataset_check.py

#









#


# ================================================================

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import rasterio
import laspy


# ================================================================

# ================================================================

DATA_ROOT = Path(
    r"G:\PhyMaize"
)

IMAGE_ROOT = (
    DATA_ROOT / "image data"
)

FEATURE_FILE = (
    DATA_ROOT / "features" / "features.csv"
)

LABEL_FILE = (
    DATA_ROOT / "annotations" / "lables.csv"
)

TRAIN_SPLIT = (
    DATA_ROOT / "meta data" / "splits" / "train.csv"
)

VAL_SPLIT = (
    DATA_ROOT / "meta data" / "splits" / "validation.csv"
)

TEST_SPLIT = (
    DATA_ROOT / "meta data" / "splits" / "test.csv"
)

# ---------------------------------------------------------------

# ---------------------------------------------------------------

MIN_POINT_COUNT = 256


# ================================================================

# ================================================================

RASTER_SUFFIXES = [

    "Green.tif",

    "Red.tif",

    "RedEdge.tif",

    "NIR.tif",

    "Thermal.tif"
]

POINTCLOUD_SUFFIX = (
    "PointCloud.las"
)


# ================================================================

# ================================================================

issues = {

    "label_invalid": [],

    "feature_invalid": [],

    "missing_directory": [],

    "missing_raster": [],

    "missing_pointcloud": [],

    "raster_invalid": [],

    "pointcloud_invalid": [],

    "low_point_count": [],

    "duplicate_sample": [],

}


# ================================================================

# ================================================================

def print_title(title):
    """Perform print title."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ================================================================

# ================================================================

def get_sample_file(
    sample_dir,
    sample_id,
    suffix
):
    """Retrieve or load sample file for subsequent analysis."""
    sample_dir = Path(
        sample_dir
    )
    expected_name = (
        f"{sample_id}_{suffix}"
    )
    expected_path = (
        sample_dir / expected_name
    )
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    if expected_path.is_file():
        return expected_path
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    target = (
        expected_name.lower()
    )
    if sample_dir.exists():
        for path in sample_dir.iterdir():
            if (
                path.is_file()
                and
                path.name.lower() == target
            ):
                return path
    return None


# ================================================================

# ================================================================

def load_csv(
    path,
    name
):
    """Retrieve or load csv for subsequent analysis."""
    print(
        f"\n[{name}] Loading:"
    )
    print(
        f"    {path}"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"找不到文件：{path}"
        )
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    try:
        df = pd.read_csv(
            path,
            encoding="utf-8-sig"
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            encoding="utf-8"
        )
    print(
        f"    Shape: {df.shape}"
    )
    print(
        f"    Columns: "
        f"{list(df.columns)}"
    )
    return df


# ================================================================

# ================================================================

def check_labels():
    """Validate labels."""
    print_title(
        "1. Label Data Quality Check"
    )
    df = load_csv(
        LABEL_FILE,
        "Labels"
    )
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    if "Sample_ID" not in df.columns:
        raise ValueError(
            "labels.csv 中不存在 Sample_ID 列"
        )
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    if "Pn" not in df.columns:
        raise ValueError(
            "labels.csv 中不存在 Pn 列"
        )
    sample_ids = (
        df["Sample_ID"]
        .astype(str)
        .str.strip()
    )
    pn = pd.to_numeric(
        df["Pn"],
        errors="coerce"
    )
    # ------------------------------------------------------------
    # NaN
    # ------------------------------------------------------------
    nan_mask = pn.isna()
    nan_count = int(
        nan_mask.sum()
    )
    # ------------------------------------------------------------
    # Inf
    # ------------------------------------------------------------
    inf_mask = (
        np.isinf(
            pn.to_numpy(
                dtype=np.float64
            )
        )
    )
    inf_count = int(
        inf_mask.sum()
    )
    print(
        f"\nPn statistics:"
    )
    print(
        f"    Samples : {len(df)}"
    )
    print(
        f"    NaN     : {nan_count}"
    )
    print(
        f"    Inf     : {inf_count}"
    )
    if nan_count > 0:
        bad_ids = (
            sample_ids[nan_mask]
            .tolist()
        )
        print(
            "\n[ERROR] Pn contains NaN:"
        )
        for sid in bad_ids:
            print(
                f"    {sid}"
            )
            issues[
                "label_invalid"
            ].append(
                sid
            )
    if inf_count > 0:
        bad_ids = (
            sample_ids[inf_mask]
            .tolist()
        )
        print(
            "\n[ERROR] Pn contains Inf:"
        )
        for sid in bad_ids:
            print(
                f"    {sid}"
            )
            issues[
                "label_invalid"
            ].append(
                sid
            )
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    valid_pn = pn[
        np.isfinite(
            pn.to_numpy(
                dtype=np.float64
            )
        )
    ]
    if len(valid_pn) > 0:
        print(
            f"\nValid Pn statistics:"
        )
        print(
            f"    Min  : "
            f"{valid_pn.min():.4f}"
        )
        print(
            f"    Max  : "
            f"{valid_pn.max():.4f}"
        )
        print(
            f"    Mean : "
            f"{valid_pn.mean():.4f}"
        )
        print(
            f"    Std  : "
            f"{valid_pn.std():.4f}"
        )
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    duplicated = (
        sample_ids[
            sample_ids.duplicated(
                keep=False
            )
        ]
    )
    if len(duplicated) > 0:
        duplicated_ids = sorted(
            set(
                duplicated.tolist()
            )
        )
        print(
            "\n[ERROR] Duplicate Sample_ID:"
        )
        for sid in duplicated_ids:
            print(
                f"    {sid}"
            )
            issues[
                "duplicate_sample"
            ].append(
                sid
            )
    return df


# ================================================================

# ================================================================

def check_features():
    """Validate features."""
    print_title(
        "2. Feature Data Quality Check"
    )
    df = load_csv(
        FEATURE_FILE,
        "Features"
    )
    if "Sample_ID" not in df.columns:
        raise ValueError(
            "features.csv 中不存在 Sample_ID 列"
        )
    feature_columns = [
        col
        for col in df.columns
        if col != "Sample_ID"
    ]
    print(
        f"\nFeature dimension:"
    )
    print(
        f"    {len(feature_columns)}"
    )
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    numeric_df = df[
        feature_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )
    # ============================================================
    # NaN
    # ============================================================
    nan_mask = (
        numeric_df.isna()
    )
    nan_total = int(
        nan_mask.sum().sum()
    )
    print(
        f"\nFeature NaN:"
    )
    print(
        f"    Total NaN values: "
        f"{nan_total}"
    )
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    nan_by_feature = (
        nan_mask.sum()
        .sort_values(
            ascending=False
        )
    )
    nan_by_feature = (
        nan_by_feature[
            nan_by_feature > 0
        ]
    )
    if len(nan_by_feature) > 0:
        print(
            "\nFeatures containing NaN:"
        )
        for feature, count in (
            nan_by_feature.items()
        ):
            print(
                f"    {feature:<35}"
                f"{int(count)}"
            )
    # ============================================================
    # Inf
    # ============================================================
    numeric_array = (
        numeric_df.to_numpy(
            dtype=np.float64
        )
    )
    inf_mask = np.isinf(
        numeric_array
    )
    inf_total = int(
        inf_mask.sum()
    )
    print(
        f"\nFeature Inf:"
    )
    print(
        f"    Total Inf values: "
        f"{inf_total}"
    )
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    if inf_total > 0:
        inf_by_feature = (
            pd.Series(
                inf_mask.sum(axis=0),
                index=feature_columns
            )
            .sort_values(
                ascending=False
            )
        )
        inf_by_feature = (
            inf_by_feature[
                inf_by_feature > 0
            ]
        )
        print(
            "\nFeatures containing Inf:"
        )
        for feature, count in (
            inf_by_feature.items()
        ):
            print(
                f"    {feature:<35}"
                f"{int(count)}"
            )
    # ============================================================
    # ============================================================
    invalid_rows = (
        ~np.isfinite(
            numeric_array
        ).all(
            axis=1
        )
    )
    invalid_indices = np.where(
        invalid_rows
    )[0]
    sample_ids = (
        df["Sample_ID"]
        .astype(str)
        .str.strip()
        .tolist()
    )
    if len(invalid_indices) > 0:
        print(
            "\n[ERROR] Samples with "
            "NaN / Inf features:"
        )
        for idx in invalid_indices:
            sid = sample_ids[idx]
            bad_features = []
            for j, feature in enumerate(
                feature_columns
            ):
                value = numeric_array[
                    idx,
                    j
                ]
                if not np.isfinite(value):
                    bad_features.append(
                        feature
                    )
            print(
                f"\n    Sample_ID: {sid}"
            )
            print(
                f"    Invalid features:"
            )
            for feature in bad_features:
                print(
                    f"        {feature}"
                )
            issues[
                "feature_invalid"
            ].append(
                sid
            )
    else:
        print(
            "\n[OK] All feature values "
            "are finite."
        )
    # ============================================================
    # Feature statistics
    # ============================================================
    finite_values = (
        numeric_array[
            np.isfinite(
                numeric_array
            )
        ]
    )
    if len(finite_values) > 0:
        print(
            "\nOverall feature value range:"
        )
        print(
            f"    Min: "
            f"{finite_values.min():.6f}"
        )
        print(
            f"    Max: "
            f"{finite_values.max():.6f}"
        )
    return df, feature_columns


# ================================================================

# ================================================================

def check_splits():
    """Validate splits."""
    print_title(
        "3. Dataset Split Check"
    )
    train_df = load_csv(
        TRAIN_SPLIT,
        "Train"
    )
    val_df = load_csv(
        VAL_SPLIT,
        "Validation"
    )
    test_df = load_csv(
        TEST_SPLIT,
        "Test"
    )
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    train_ids = set(
        train_df[
            "Sample_ID"
        ]
        .astype(str)
        .str.strip()
    )
    val_ids = set(
        val_df[
            "Sample_ID"
        ]
        .astype(str)
        .str.strip()
    )
    test_ids = set(
        test_df[
            "Sample_ID"
        ]
        .astype(str)
        .str.strip()
    )
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    print(
        "\nSplit summary:"
    )
    print(
        f"    Train      : "
        f"{len(train_ids)}"
    )
    print(
        f"    Validation : "
        f"{len(val_ids)}"
    )
    print(
        f"    Test       : "
        f"{len(test_ids)}"
    )
    print(
        f"    Total      : "
        f"{len(train_ids | val_ids | test_ids)}"
    )
    # ============================================================
    # ============================================================
    train_val = (
        train_ids & val_ids
    )
    train_test = (
        train_ids & test_ids
    )
    val_test = (
        val_ids & test_ids
    )
    if len(train_val) > 0:
        print(
            "\n[ERROR] Train / Validation overlap:"
        )
        print(
            sorted(train_val)
        )
    if len(train_test) > 0:
        print(
            "\n[ERROR] Train / Test overlap:"
        )
        print(
            sorted(train_test)
        )
    if len(val_test) > 0:
        print(
            "\n[ERROR] Validation / Test overlap:"
        )
        print(
            sorted(val_test)
        )
    if (
        len(train_val) == 0
        and
        len(train_test) == 0
        and
        len(val_test) == 0
    ):
        print(
            "\n[OK] No split overlap detected."
        )
    return (
        train_ids,
        val_ids,
        test_ids
    )


# ================================================================

# ================================================================

def check_raster(
    path,
    sample_id,
    modality
):
    """Validate raster."""
    try:
        with rasterio.open(
            path
        ) as src:
            data = src.read(
                1
            )
            shape = data.shape
            dtype = data.dtype
            # ----------------------------------------------------
            # ----------------------------------------------------
            if np.issubdtype(
                data.dtype,
                np.number
            ):
                finite_mask = np.isfinite(
                    data
                )
                invalid_count = int(
                    (~finite_mask).sum()
                )
                if invalid_count > 0:
                    issues[
                        "raster_invalid"
                    ].append({
                        "Sample_ID":
                            sample_id,
                        "Modality":
                            modality,
                        "File":
                            str(path),
                        "Invalid":
                            invalid_count
                    })
                    print(
                        f"\n[ERROR] Raster invalid:"
                    )
                    print(
                        f"    Sample_ID : "
                        f"{sample_id}"
                    )
                    print(
                        f"    Modality  : "
                        f"{modality}"
                    )
                    print(
                        f"    File      : "
                        f"{path}"
                    )
                    print(
                        f"    Shape     : "
                        f"{shape}"
                    )
                    print(
                        f"    NaN/Inf   : "
                        f"{invalid_count}"
                    )
            else:
                print(
                    f"\n[WARNING] "
                    f"Non-numeric raster:"
                )
                print(
                    f"    {path}"
                )
    except Exception as e:
        print(
            f"\n[ERROR] Cannot read raster:"
        )
        print(
            f"    Sample_ID : {sample_id}"
        )
        print(
            f"    Modality  : {modality}"
        )
        print(
            f"    File      : {path}"
        )
        print(
            f"    Error     : {e}"
        )
        issues[
            "raster_invalid"
        ].append({
            "Sample_ID":
                sample_id,
            "Modality":
                modality,
            "File":
                str(path),
            "Invalid":
                "READ_ERROR"
        })


# ================================================================

# ================================================================

def check_pointcloud(
    path,
    sample_id
):
    """Validate pointcloud."""
    try:
        # --------------------------------------------------------
        # --------------------------------------------------------
        las = laspy.read(
            path
        )
        point_count = len(
            las.points
        )
        # --------------------------------------------------------
        # --------------------------------------------------------
        if (
            point_count
            <
            MIN_POINT_COUNT
        ):
            print(
                f"\n[WARNING] "
                f"Low point count:"
            )
            print(
                f"    Sample_ID : "
                f"{sample_id}"
            )
            print(
                f"    Points    : "
                f"{point_count}"
            )
            print(
                f"    Threshold : "
                f"{MIN_POINT_COUNT}"
            )
            issues[
                "low_point_count"
            ].append({
                "Sample_ID":
                    sample_id,
                "Point_Count":
                    point_count
            })
        # --------------------------------------------------------
        # XYZ
        # --------------------------------------------------------
        x = np.asarray(
            las.x
        )
        y = np.asarray(
            las.y
        )
        z = np.asarray(
            las.z
        )
        xyz = np.column_stack(
            [
                x,
                y,
                z
            ]
        )
        finite_mask = np.isfinite(
            xyz
        )
        invalid_count = int(
            (~finite_mask).sum()
        )
        if invalid_count > 0:
            print(
                f"\n[ERROR] "
                f"Point cloud contains "
                f"NaN/Inf:"
            )
            print(
                f"    Sample_ID : "
                f"{sample_id}"
            )
            print(
                f"    File      : "
                f"{path}"
            )
            print(
                f"    Invalid XYZ values: "
                f"{invalid_count}"
            )
            issues[
                "pointcloud_invalid"
            ].append({
                "Sample_ID":
                    sample_id,
                "File":
                    str(path),
                "Invalid":
                    invalid_count
            })
        # --------------------------------------------------------
        # XYZ statistics
        # --------------------------------------------------------
        if point_count > 0:
            valid_xyz = xyz[
                np.isfinite(
                    xyz
                ).all(
                    axis=1
                )
            ]
            if len(valid_xyz) > 0:
                xyz_min = (
                    valid_xyz.min(
                        axis=0
                    )
                )
                xyz_max = (
                    valid_xyz.max(
                        axis=0
                    )
                )
    except Exception as e:
        print(
            f"\n[ERROR] Cannot read LAS:"
        )
        print(
            f"    Sample_ID : "
            f"{sample_id}"
        )
        print(
            f"    File      : "
            f"{path}"
        )
        print(
            f"    Error     : "
            f"{e}"
        )
        issues[
            "pointcloud_invalid"
        ].append({
            "Sample_ID":
                sample_id,
            "File":
                str(path),
            "Invalid":
                "READ_ERROR"
        })


# ================================================================

# ================================================================

def check_sample_files(
    all_sample_ids
):
    """Validate sample files."""
    print_title(
        "4. Image / Thermal / Point Cloud Check"
    )
    total = len(
        all_sample_ids
    )
    print(
        f"Total samples to check: "
        f"{total}"
    )
    for index, sample_id in enumerate(
        all_sample_ids,
        start=1
    ):
        sample_dir = (
            IMAGE_ROOT / sample_id
        )
        # --------------------------------------------------------
        # --------------------------------------------------------
        if (
            index == 1
            or
            index % 100 == 0
            or
            index == total
        ):
            print(
                f"\nChecking "
                f"[{index}/{total}] "
                f"{sample_id}"
            )
        # ========================================================
        # Sample directory
        # ========================================================
        if not sample_dir.exists():
            print(
                f"\n[ERROR] Missing directory:"
            )
            print(
                f"    {sample_dir}"
            )
            issues[
                "missing_directory"
            ].append(
                sample_id
            )
            continue
        # ========================================================
        # Rasters
        # ========================================================
        for suffix in RASTER_SUFFIXES:
            path = get_sample_file(
                sample_dir,
                sample_id,
                suffix
            )
            if path is None:
                print(
                    f"\n[ERROR] Missing raster:"
                )
                print(
                    f"    Sample_ID : "
                    f"{sample_id}"
                )
                print(
                    f"    Expected  : "
                    f"{sample_id}_{suffix}"
                )
                issues[
                    "missing_raster"
                ].append({
                    "Sample_ID":
                        sample_id,
                    "File":
                        f"{sample_id}_{suffix}"
                })
            else:
                check_raster(
                    path,
                    sample_id,
                    suffix.replace(
                        ".tif",
                        ""
                    )
                )
        # ========================================================
        # Point cloud
        # ========================================================
        pointcloud_path = get_sample_file(
            sample_dir,
            sample_id,
            POINTCLOUD_SUFFIX
        )
        if pointcloud_path is None:
            print(
                f"\n[WARNING] Missing point cloud:"
            )
            print(
                f"    Sample_ID : "
                f"{sample_id}"
            )
            print(
                f"    Expected  : "
                f"{sample_id}_"
                f"{POINTCLOUD_SUFFIX}"
            )
            issues[
                "missing_pointcloud"
            ].append(
                sample_id
            )
        else:
            check_pointcloud(
                pointcloud_path,
                sample_id
            )


# ================================================================

# ================================================================

def check_dataset_consistency(
    label_df,
    feature_df,
    train_ids,
    val_ids,
    test_ids
):
    """Validate dataset consistency."""
    print_title(
        "5. Dataset Consistency Check"
    )
    label_ids = set(
        label_df[
            "Sample_ID"
        ]
        .astype(str)
        .str.strip()
    )
    feature_ids = set(
        feature_df[
            "Sample_ID"
        ]
        .astype(str)
        .str.strip()
    )
    split_ids = (
        train_ids
        |
        val_ids
        |
        test_ids
    )
    # ============================================================
    # Label vs Feature
    # ============================================================
    label_only = (
        label_ids
        -
        feature_ids
    )
    feature_only = (
        feature_ids
        -
        label_ids
    )
    if len(label_only) > 0:
        print(
            "\n[ERROR] IDs in labels "
            "but not features:"
        )
        print(
            sorted(label_only)
        )
    if len(feature_only) > 0:
        print(
            "\n[ERROR] IDs in features "
            "but not labels:"
        )
        print(
            sorted(feature_only)
        )
    if (
        len(label_only) == 0
        and
        len(feature_only) == 0
    ):
        print(
            "\n[OK] Labels and features "
            "have identical Sample_ID sets."
        )
    # ============================================================
    # Labels vs Splits
    # ============================================================
    label_not_split = (
        label_ids
        -
        split_ids
    )
    split_not_label = (
        split_ids
        -
        label_ids
    )
    if len(label_not_split) > 0:
        print(
            "\n[WARNING] Labels not included "
            "in train/val/test:"
        )
        print(
            sorted(label_not_split)
        )
    if len(split_not_label) > 0:
        print(
            "\n[ERROR] Split IDs without labels:"
        )
        print(
            sorted(split_not_label)
        )
    if (
        len(label_not_split) == 0
        and
        len(split_not_label) == 0
    ):
        print(
            "\n[OK] Labels and splits "
            "are fully consistent."
        )


# ================================================================

# ================================================================

def print_final_report():
    """Perform print final report."""
    print_title(
        "FINAL DATASET QUALITY REPORT"
    )
    # ------------------------------------------------------------
    # Label
    # ------------------------------------------------------------
    print(
        "\n[Labels]"
    )
    print(
        f"    Invalid samples : "
        f"{len(set(issues['label_invalid']))}"
    )
    # ------------------------------------------------------------
    # Features
    # ------------------------------------------------------------
    feature_invalid_ids = set(
        issues[
            "feature_invalid"
        ]
    )
    print(
        "\n[Features]"
    )
    print(
        f"    Invalid samples : "
        f"{len(feature_invalid_ids)}"
    )
    if len(feature_invalid_ids) > 0:
        print(
            "    Sample_ID:"
        )
        for sid in sorted(
            feature_invalid_ids
        ):
            print(
                f"        {sid}"
            )
    # ------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------
    print(
        "\n[Sample Directory]"
    )
    print(
        f"    Missing directory : "
        f"{len(issues['missing_directory'])}"
    )
    # ------------------------------------------------------------
    # Raster
    # ------------------------------------------------------------
    print(
        "\n[Raster]"
    )
    print(
        f"    Missing files : "
        f"{len(issues['missing_raster'])}"
    )
    print(
        f"    Invalid files : "
        f"{len(issues['raster_invalid'])}"
    )
    # ------------------------------------------------------------
    # Point cloud
    # ------------------------------------------------------------
    print(
        "\n[Point Cloud]"
    )
    print(
        f"    Missing files : "
        f"{len(issues['missing_pointcloud'])}"
    )
    print(
        f"    Invalid files : "
        f"{len(issues['pointcloud_invalid'])}"
    )
    print(
        f"    Low point-count samples : "
        f"{len(issues['low_point_count'])}"
    )
    # ============================================================
    # Missing point cloud IDs
    # ============================================================
    if len(
        issues["missing_pointcloud"]
    ) > 0:
        print(
            "\nMissing PointCloud Sample_ID:"
        )
        for sid in sorted(
            issues[
                "missing_pointcloud"
            ]
        ):
            print(
                f"    {sid}"
            )
    # ============================================================
    # Low point count
    # ============================================================
    if len(
        issues["low_point_count"]
    ) > 0:
        print(
            "\nLow Point Count:"
        )
        for item in sorted(
            issues["low_point_count"],
            key=lambda x: x["Point_Count"]
        ):
            print(
                f"    "
                f"{item['Sample_ID']:<12}"
                f"Points = "
                f"{item['Point_Count']}"
            )
    # ============================================================
    # Raster invalid
    # ============================================================
    if len(
        issues["raster_invalid"]
    ) > 0:
        print(
            "\nInvalid Raster Files:"
        )
        for item in issues[
            "raster_invalid"
        ]:
            print(
                f"    "
                f"{item['Sample_ID']:<12}"
                f"{item['Modality']:<12}"
                f"Invalid = "
                f"{item['Invalid']}"
            )
    # ============================================================
    # Point cloud invalid
    # ============================================================
    if len(
        issues["pointcloud_invalid"]
    ) > 0:
        print(
            "\nInvalid Point Cloud Files:"
        )
        for item in issues[
            "pointcloud_invalid"
        ]:
            print(
                f"    "
                f"{item['Sample_ID']:<12}"
                f"Invalid = "
                f"{item['Invalid']}"
            )
    # ============================================================
    # Overall status
    # ============================================================
    has_fatal_issue = (
        len(
            issues["label_invalid"]
        ) > 0
        or
        len(
            issues["feature_invalid"]
        ) > 0
        or
        len(
            issues["missing_directory"]
        ) > 0
        or
        len(
            issues["missing_raster"]
        ) > 0
        or
        len(
            issues["raster_invalid"]
        ) > 0
        or
        len(
            issues["pointcloud_invalid"]
        ) > 0
    )
    print()
    print("=" * 70)
    if has_fatal_issue:
        print(
            "RESULT: DATASET HAS ERRORS"
        )
        print(
            "请根据上面的 Sample_ID "
            "和具体模态进一步处理。"
        )
    else:
        print(
            "RESULT: NO FATAL DATA ERRORS"
        )
        if (
            len(
                issues[
                    "missing_pointcloud"
                ]
            ) > 0
            or
            len(
                issues[
                    "low_point_count"
                ]
            ) > 0
        ):
            print(
                "存在缺失/低点数点云，"
                "但这些问题不会被本检查脚本修改。"
            )
        else:
            print(
                "All checked data are valid."
            )
    print("=" * 70)


# ================================================================
# 15. Main
# ================================================================

def main():
    """Run the main analysis workflow."""
    print()
    print("=" * 70)
    print(
        "             PhyMaize Dataset Check"
    )
    print("=" * 70)
    print(
        f"\nData root:"
    )
    print(
        f"    {DATA_ROOT}"
    )
    print(
        "\n注意："
    )
    print(
        "    本程序只读取数据。"
    )
    print(
        "    不会删除、修改或覆盖任何原始文件。"
    )
    # ============================================================
    # Labels
    # ============================================================
    label_df = check_labels()
    # ============================================================
    # Features
    # ============================================================
    feature_df, feature_columns = (
        check_features()
    )
    # ============================================================
    # Splits
    # ============================================================
    (
        train_ids,
        val_ids,
        test_ids
    ) = check_splits()
    # ============================================================
    # Dataset consistency
    # ============================================================
    check_dataset_consistency(
        label_df,
        feature_df,
        train_ids,
        val_ids,
        test_ids
    )
    # ============================================================
    # ============================================================
    all_sample_ids = sorted(
        train_ids
        |
        val_ids
        |
        test_ids
    )
    # ============================================================
    # ============================================================
    check_sample_files(
        all_sample_ids
    )
    # ============================================================
    # Final report
    # ============================================================
    print_final_report()


# ================================================================
# Entry
# ================================================================

if __name__ == "__main__":

    main()