"""Construct a consolidated dataset from photosynthesis and multimodal feature files."""
import os
import shutil
import re


# ============================================================

# ============================================================


MULTISPECTRAL_ROOT = r"G:\2025公主岭无人机数据\M3M\M3M-V12"


THERMAL_DIR = r"G:\2025公主岭无人机数据\M3T\M3T-V12"


POINT_CLOUD_DIR = r"G:\2025公主岭无人机数据\M350\V12-M350\final"


OUTPUT_ROOT = r"G:\PhyMaize\image data"


# ============================================================

# ============================================================


# START_INDEX = 1      -> PM_0001
# START_INDEX = 325    -> PM_0325
START_INDEX = 2357


# ============================================================

# ============================================================

BAND_FOLDERS = {
    "Green": "Green",
    "Red": "Red",
    "RedEdge": "RedEdge",
    "NIR": "NIR"
}


# ============================================================

# ============================================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ============================================================

# ============================================================

def get_files(directory, extensions):
    """
    获取指定目录下指定扩展名的文件。
    """
    files = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            if filename.lower().endswith(extensions):
                files.append(file_path)
    return files


# ============================================================

# ============================================================

def normalize_sample_id(filename):
    """
    统一样本名称。

    示例：
        1-1.tif  -> 1_1
        1_1.tif  -> 1_1
        1-1.las  -> 1_1
        1_1.las  -> 1_1
    """
    name = os.path.splitext(filename)[0]
    name = name.replace("-", "_")
    return name


# ============================================================

# ============================================================

def build_file_index(directory, extensions):
    """
    建立：

        样本ID -> 文件路径

    的索引。
    """
    file_index = {}
    files = get_files(directory, extensions)
    for file_path in files:
        filename = os.path.basename(file_path)
        sample_id = normalize_sample_id(filename)
        if sample_id in file_index:
            print(f"\n警告：发现重复样本 ID：{sample_id}")
            print(f"已有文件：{file_index[sample_id]}")
            print(f"新文件：{file_path}")
        file_index[sample_id] = file_path
    return file_index


# ============================================================

# ============================================================

print("=" * 70)
print("正在建立多光谱文件索引...")
print("=" * 70)


multispectral_index = {}


for band_name, folder_name in BAND_FOLDERS.items():

    band_dir = os.path.join(
        MULTISPECTRAL_ROOT,
        folder_name
    )

    print(f"\n正在读取 {band_name} 波段...")

    print(f"路径：{band_dir}")

    if not os.path.exists(band_dir):

        raise FileNotFoundError(
            f"未找到文件夹：{band_dir}"
        )

    multispectral_index[band_name] = build_file_index(
        band_dir,
        (".tif", ".tiff")
    )

    print(
        f"{band_name} 文件数量："
        f"{len(multispectral_index[band_name])}"
    )


# ============================================================

# ============================================================

print("\n" + "=" * 70)
print("正在建立热红外文件索引...")
print("=" * 70)


if not os.path.exists(THERMAL_DIR):

    raise FileNotFoundError(
        f"未找到热红外文件夹：{THERMAL_DIR}"
    )


thermal_index = build_file_index(
    THERMAL_DIR,
    (".tif", ".tiff")
)


print(
    f"热红外文件数量："
    f"{len(thermal_index)}"
)


# ============================================================

# ============================================================

print("\n" + "=" * 70)
print("正在建立LAS点云文件索引...")
print("=" * 70)


if not os.path.exists(POINT_CLOUD_DIR):

    raise FileNotFoundError(
        f"未找到点云文件夹：{POINT_CLOUD_DIR}"
    )


point_cloud_index = build_file_index(
    POINT_CLOUD_DIR,
    (".las", ".laz")
)


print(
    f"点云文件数量："
    f"{len(point_cloud_index)}"
)


# ============================================================

# ============================================================


all_sample_ids = set(
    multispectral_index["Green"].keys()
)


# ============================================================

# ============================================================

def natural_sort_key(sample_id):
    """
    对样本 ID 进行自然排序。

    示例：

        1_1
        1_2
        1_10
        2_1
    """
    parts = re.split(
        r"(\d+)",
        sample_id
    )
    return [
        int(part)
        if part.isdigit()
        else part.lower()
        for part in parts
    ]


sorted_sample_ids = sorted(
    all_sample_ids,
    key=natural_sort_key
)


# ============================================================

# ============================================================

print("\n" + "=" * 70)
print("检查样本完整性...")
print("=" * 70)



valid_samples = []


invalid_samples = []


missing_pointcloud_samples = []


for sample_id in sorted_sample_ids:

    missing_required_files = []


    # --------------------------------------------------------

    # --------------------------------------------------------

    for band_name in BAND_FOLDERS.keys():

        if sample_id not in multispectral_index[band_name]:

            missing_required_files.append(
                band_name
            )


    # --------------------------------------------------------

    # --------------------------------------------------------

    if sample_id not in thermal_index:

        missing_required_files.append(
            "Thermal"
        )


    # --------------------------------------------------------


    # --------------------------------------------------------

    if sample_id not in point_cloud_index:

        missing_pointcloud_samples.append(
            sample_id
        )


    # --------------------------------------------------------

    # --------------------------------------------------------




    if len(missing_required_files) == 0:

        valid_samples.append(
            sample_id
        )

    else:

        invalid_samples.append(

            (
                sample_id,
                missing_required_files
            )

        )


# ============================================================

# ============================================================

print("\n" + "=" * 70)
print("样本完整性检查结果")
print("=" * 70)


print(
    f"候选样本总数："
    f"{len(sorted_sample_ids)}"
)


print(
    f"可生成样本数量："
    f"{len(valid_samples)}"
)


print(
    f"缺少必要数据的样本数量："
    f"{len(invalid_samples)}"
)


print(
    f"缺失点云的样本数量："
    f"{len(missing_pointcloud_samples)}"
)


# ------------------------------------------------------------

# ------------------------------------------------------------

if len(invalid_samples) > 0:

    print("\n以下样本缺少必要数据：")

    for sample_id, missing_files in invalid_samples:

        print(
            f"{sample_id} "
            f"缺少：{', '.join(missing_files)}"
        )


# ------------------------------------------------------------

# ------------------------------------------------------------

if len(missing_pointcloud_samples) > 0:

    print("\n以下样本缺失点云，但仍会生成样本文件夹：")

    for sample_id in missing_pointcloud_samples:

        print(
            f"  {sample_id}"
        )


# ============================================================

# ============================================================

print("\n" + "=" * 70)
print("开始复制和整理样本...")
print("=" * 70)


current_index = START_INDEX


for sample_id in valid_samples:


    # ========================================================

    # ========================================================

    new_sample_id = (
        f"PM_{current_index:04d}"
    )


    # ========================================================

    # ========================================================

    sample_output_dir = os.path.join(
        OUTPUT_ROOT,
        new_sample_id
    )


    os.makedirs(
        sample_output_dir,
        exist_ok=True
    )


    print("\n" + "-" * 70)

    print(
        f"[{new_sample_id}] "
        f"原始样本：{sample_id}"
    )


    # ========================================================

    # ========================================================

    source_file = multispectral_index[
        "Green"
    ][sample_id]


    destination_file = os.path.join(
        sample_output_dir,
        f"{new_sample_id}_Green.tif"
    )


    shutil.copy2(
        source_file,
        destination_file
    )


    print("  ✓ Green")


    # ========================================================

    # ========================================================

    source_file = multispectral_index[
        "Red"
    ][sample_id]


    destination_file = os.path.join(
        sample_output_dir,
        f"{new_sample_id}_Red.tif"
    )


    shutil.copy2(
        source_file,
        destination_file
    )


    print("  ✓ Red")


    # ========================================================

    # ========================================================

    source_file = multispectral_index[
        "RedEdge"
    ][sample_id]


    destination_file = os.path.join(
        sample_output_dir,
        f"{new_sample_id}_RedEdge.tif"
    )


    shutil.copy2(
        source_file,
        destination_file
    )


    print("  ✓ RedEdge")


    # ========================================================

    # ========================================================

    source_file = multispectral_index[
        "NIR"
    ][sample_id]


    destination_file = os.path.join(
        sample_output_dir,
        f"{new_sample_id}_NIR.tif"
    )


    shutil.copy2(
        source_file,
        destination_file
    )


    print("  ✓ NIR")


    # ========================================================

    # ========================================================

    source_file = thermal_index[
        sample_id
    ]


    destination_file = os.path.join(
        sample_output_dir,
        f"{new_sample_id}_Thermal.tif"
    )


    shutil.copy2(
        source_file,
        destination_file
    )


    print("  ✓ Thermal")


    # ========================================================

    # ========================================================


    if sample_id in point_cloud_index:


        source_file = point_cloud_index[
            sample_id
        ]



        pointcloud_extension = os.path.splitext(
            source_file
        )[1]


        destination_file = os.path.join(

            sample_output_dir,

            f"{new_sample_id}_PointCloud"
            f"{pointcloud_extension}"

        )


        shutil.copy2(
            source_file,
            destination_file
        )


        print("  ✓ PointCloud")


    else:

        print(
            "  ⚠ PointCloud 缺失（允许缺失）"
        )


    # ========================================================

    # ========================================================

    current_index += 1


# ============================================================

# ============================================================

print("\n" + "=" * 70)
print("数据整理完成！")
print("=" * 70)


print(
    f"成功生成样本数量："
    f"{len(valid_samples)}"
)


print(
    f"其中完整6模态样本数量："
    f"{len(valid_samples) - len(missing_pointcloud_samples)}"
)


print(
    f"缺失点云但保留的样本数量："
    f"{len(missing_pointcloud_samples)}"
)


print(
    f"输出目录："
    f"{OUTPUT_ROOT}"
)


if len(valid_samples) > 0:

    print(
        f"起始编号："
        f"PM_{START_INDEX:04d}"
    )


    print(
        f"结束编号："
        f"PM_{current_index - 1:04d}"
    )


print("=" * 70)