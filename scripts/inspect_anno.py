
from pathlib import Path
import h5py

mat_path = Path(r"../data/raw/train/digitStruct.mat")

with h5py.File(mat_path, "r") as f:
    print("Top-level keys:", list(f.keys()))

    digit_struct = f["digitStruct"]
    print("digitStruct keys:", list(digit_struct.keys()))

    names = digit_struct["name"]
    boxes = digit_struct["bbox"]

    print("Number of annotated images:", len(names))
    print("Number of bounding-box entries:", len(boxes))

    print("\nFirst image reference:", names[0, 0])
    print("First bounding-box reference:", boxes[0, 0])
