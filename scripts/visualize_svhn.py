
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle

# Change this to your training directory
DATA_DIR = Path(r"../data/raw/train/")
MAT_FILE = DATA_DIR / "digitStruct.mat"


def read_string(h5_file, ref):
    """Read an image filename through its HDF5 reference."""
    return "".join(
        chr(c[0]) for c in h5_file[ref][()]
    )


def read_value(h5_file, ref):
    """Read a scalar or array stored through an HDF5 reference."""
    data = np.asarray(h5_file[ref][()]).reshape(-1)
    return data.astype(float)


def read_field(h5_file, bbox_ref, field):
    """Read a bounding-box field from an SVHN HDF5 annotation."""
    dataset = h5_file[bbox_ref][field]
    raw = dataset[()]

    # Some fields contain numeric values directly.
    # Others contain references to datasets holding the values.
    ref_type = h5py.check_dtype(ref=dataset.dtype)

    if ref_type is None:
        return np.asarray(raw, dtype=float).reshape(-1)

    values = []

    for ref in np.asarray(raw).reshape(-1):
        value = np.asarray(h5_file[ref][()]).reshape(-1)
        values.append(float(value[0]))

    return np.asarray(values, dtype=float)


def load_annotation(h5_file, index):
    """Return the filename and all digit annotations for one image."""
    struct = h5_file["digitStruct"]
    name_ref = struct["name"][index, 0]
    bbox_ref = struct["bbox"][index, 0]

    filename = read_string(h5_file, name_ref)

    annotations = {}
    for field in ("label", "left", "top", "width", "height"):
        annotations[field] = read_field(h5_file, bbox_ref, field)

    return filename, annotations


def show_sample(index):
    with h5py.File(MAT_FILE, "r") as f:
        filename, ann = load_annotation(f, index)

    image_path = DATA_DIR / filename
    image = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)

    for label, left, top, width, height in zip(
        ann["label"],
        ann["left"],
        ann["top"],
        ann["width"],
        ann["height"],
    ):
        digit = 0 if int(label) == 10 else int(label)

        # MATLAB annotations use 1-based pixel coordinates.
        x = left - 1
        y = top - 1

        ax.add_patch(Rectangle(
            (x, y), width, height,
            fill=False, edgecolor="lime", linewidth=2
        ))

        ax.text(
            x, y - 3, str(digit),
            color="black", fontsize=12, weight="bold",
            bbox=dict(facecolor="lime", alpha=0.8, pad=2)
        )

    ax.set_title(f"{filename} | Labels: {ann['label'].astype(int)}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for i in (0, 1, 2, 10, 100):
        show_sample(i)
