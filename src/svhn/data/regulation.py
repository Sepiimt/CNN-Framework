# --- Imports ---
import joblib #> Saving python lists

# --- Functions ---
def save_annotation_data(
        path: str,
        y1: list, 
        y2: list,
        x1: list, 
        x2: list, 
        labels: list, 
        name: list
        ) -> None:
    # --- Saving Data ---
    #> `compress=3` provides a great balance of speed and file size reduction
    joblib.dump(y1, f"{path}y1_data.joblib", compress=3)
    joblib.dump(y2, f"{path}y2_data.joblib", compress=3)
    joblib.dump(x1, f"{path}x1_data.joblib", compress=3)
    joblib.dump(x2, f"{path}x2_data.joblib", compress=3)
    joblib.dump(labels, f"{path}labels_data.joblib", compress=3)
    joblib.dump(name, f"{path}names_data.joblib", compress=3)

def load_annotation_data(path: str) -> tuple[list, ...]:
    # --- Loading Data ---
    x1 = joblib.load(f"{path}x1_data.joblib")
    x2 = joblib.load(f"{path}x2_data.joblib")
    y1 = joblib.load(f"{path}y1_data.joblib")
    y2 = joblib.load(f"{path}y2_data.joblib")
    labels = joblib.load(f"{path}labels_data.joblib")
    names = joblib.load(f"{path}names_data.joblib")
    # --- Return ---
    return y1, y2, x1, x2, labels, names