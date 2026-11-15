# --- Imports ---
import random #> Generating random numbers
import numpy as np #> Handling arrays
import cv2 #> Reading image files
import matplotlib.pyplot as plt #> Plotting sample images
import matplotlib.patches as patches #> Extension to plotting

# --- Functions ---
def random_sample_visulization(
    n, path, 
    y1_ls, y2_ls, x1_ls, x2_ls,  
    labels_ls, names_ls, figsize=(12, 12)
    ) -> None:
  # --- Getting Info ---
  total_images = len(names_ls)
  num_samples = n * n
  # --- Handeling Invalid Input ---
  if num_samples > total_images:
    raise ValueError(
        f"Requested {num_samples} images ({n}x{n}), but dataset only has"
        f" {total_images} items.")
  # --- Generating n^2 Random Unique Indices ---
  random_indices = random.sample(range(total_images), num_samples)
  # --- Creating Subplots ---
  fig, axes = plt.subplots(n, n, figsize=figsize)
  axes = np.array(axes).reshape(n, n)  # Ensure 2D indexing works even if n=1
  # --- Looping Through Random Indices and Plotting ---
  for idx, png_num in enumerate(random_indices):
    row, col = divmod(idx, n)
    ax = axes[row, col]
    # --- Loading Images & Converting BGR to RGB ---
    img_path = f"{path}/{names_ls[png_num]}"
    image_data = cv2.imread(img_path)
    # --- Displaying Image ---
    if image_data is not None:
      image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
      ax.imshow(image_data)
    # --- Drawing Bounding Boxes and Labels ---
    for y1, y2, x1, x2, label in zip(
        y1_ls[png_num],
        y2_ls[png_num],
        x1_ls[png_num],
        x2_ls[png_num],
        labels_ls[png_num],):
      rect = patches.Rectangle(
          (x1, y1),
          x2 - x1,
          y2 - y1,
          linewidth=1.5,
          edgecolor="red",
          facecolor="none",)
      ax.add_patch(rect)
      ax.text(
          x1,
          y1,
          str(label),
          color="yellow",
          fontsize=8,
          bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"),)
    # --- Finalizing Each Subplot ---
    ax.axis("off")
    ax.set_title(f"ID: {png_num}", fontsize=9)
  # --- Finalizing the Entire Figure ---
  plt.tight_layout()
  plt.show()
  plt.close()

# Example usage (e.g., a 3x3 grid = 9 random images):
# visualize_random_grid(3, '../data/raw/train', x1_ls, x2_ls, y1_ls, y2_ls, labels_ls, names_ls)