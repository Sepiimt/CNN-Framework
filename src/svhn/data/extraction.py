# --- Imports ---
import h5py #> Reading `.mat` file
import numpy as np #> Handling arrays

# --- Functions ---
def _read_mat_field(file, dataset):
    if dataset.shape[0] == 1:
        return [float(dataset[0][0])]
    return [float(file[ref[0]][()].item())for ref in dataset]

def mat_data_extraction(
        path: str
        ) -> tuple[list, ...]:
    # --- Creating Containers ---
    y1, x1, height_ls, width_ls, labels_ls, name_ls = [], [], [], [], [], []
    # --- Opening File ---
    with h5py.File(f"{path}digitStruct.mat", "r") as f:
        names = f["digitStruct/name"]
        boxes = f["digitStruct/bbox"]
        # --- Iterating on Data ---
        for index_name, index_box in zip(names, boxes):
            chars = f[index_name[0]][()].ravel()
            filename = "".join(chr(int(c)) for c in chars)
            index_box_group = f[index_box[0]]
            # --- Appending Data to Container ---
            x1.append(_read_mat_field(f, index_box_group["left"]))
            y1.append(_read_mat_field(f, index_box_group["top"]))
            width_ls.append(_read_mat_field(f, index_box_group["width"]))
            height_ls.append(_read_mat_field(f, index_box_group["height"]))
            labels_ls.append(_read_mat_field(f, index_box_group["label"]))
            name_ls.append(filename)
    # --- Return ---
    return  y1, x1, height_ls, width_ls, labels_ls, name_ls

def coordinate_to_point(
        y1: list,
        x1: list,
        height_ls: list,  
        width_ls: list, 
        ) -> tuple[list, list]:
    # --- Converting Coordinates Into Points ---
    x2 = []
    for fa,sa in zip(x1, width_ls):
        x2.append(np.array(fa) + np.array(sa))
    y2 = []
    for fa,sa in zip(y1, height_ls):
        y2.append(np.array(fa) + np.array(sa))
    # --- Return ---
    return y2, x2
    
def replace_10_labels_with_0(labels_ls: list) -> list:
    # --- Replacing 10 with 0 (The zeros are labeled as 10) in Labels ---
    for i, ls in enumerate(labels_ls):
        ls = np.array(ls)
        ls[ls == 10.0] = 0
        labels_ls[i] = ls.astype(int)
    # --- Return ---
    return labels_ls

def inner_data_conversion_to_numpy(
        y1: list, 
        y2: list,
        x1: list, 
        x2: list, 
        ) -> tuple[list, ...]:
    # --- Converting Inner Indexes to Numpy Arrays ---
    for i, lsy1, lsy2, lsx1, lsx2 in zip(range(len(x1)), y1, y2, x1, x2):
        y1[i] = np.array(lsy1).ravel()
        y2[i] = np.array(lsy2).ravel()
        x1[i] = np.array(lsx1).ravel()
        x2[i] = np.array(lsx2).ravel()
    # --- Return ---
    return  y1, y2, x1, x2