# --- Imports ---
import cv2 
import numpy as np
from pathlib import Path
import os
from typeguard import typechecked

from ..utils.typing import PaddingVar, DoubleIntTuple, TripleIntTuple

import warnings
from ..utils.warnings import DirWarning

from ..utils.exceptions import PathError


# --- Functions ---
@typechecked
def letterbox(
        input_path: str | Path,
        output_path: str | Path,
        y1: list,
        y2: list,
        x1: list,
        x2: list,
        names_ls: list, #> Image names present at directory
        /,
        height: int = 250,
        width: int = 250, 
        *,
        pbt: PaddingVar = "Replicate", #> Padding Border Type
        const_pbt_value: TripleIntTuple = (0, 0, 0),
        force_dir: bool = True):
    # --- Input Validation ---
    _input_validation(
        input_path = input_path,
        output_path = output_path,
        force_dir = force_dir,
        y1 = y1, y2 = y2, 
        x1 = x1, x2 = x2,
        names_ls = names_ls)
    # --- Iterating ---
    for count, img_name in enumerate(names_ls):
        # --- Reading Image ---
        img_sample, si_height, si_width = _read_img_data(input_path, img_name)
        # --- Calculating Resize Values ---
        (res_height, res_width, y_padding, x_padding
         ) = _resize_calculation(
             height = height, width = width, 
             si_height = si_height, si_width = si_width)
        # --- Resizing ---
        img_sample = _img_resizer(img_sample, res_width, res_height)
        # --- Center Padding ---
        padded_img = _img_padder(
            img_sample = img_sample, 
            y_padding = y_padding, 
            x_padding = x_padding, pbt = pbt, 
            const_pbt_value = const_pbt_value)
        # --- Saving Processed Image ---
        _processed_img_save(
            padded_img = padded_img, 
            output_path = output_path, 
            img_name = img_name)
        # --- Resizing Annotation Values ---
        y1i, y2i, x1i, x2i = _annotation_data_adjust(
            si_height = si_height, 
            si_width = si_width,
            res_height = res_height, 
            res_width = res_width,
            y1i = y1[count], y2i = y2[count], 
            x1i = x1[count], x2i = x2[count], 
            y_padding = y_padding, 
            x_padding = x_padding)
        # --- Updating Annotation List ---
        y1, y2, x1, x2 = _annotation_list_update(
            y1 = y1, y2 = y2, 
            x1 = x1, x2 = x2,
            y1i = y1i, y2i = y2i, 
            x1i = x1i, x2i = x2i,
            count = count)
    # --- Return ---
    return y1, y2, x1, x2


def _input_validation(*args, **kwargs) -> None:
    # --- Input Path Check ---
    if not Path.is_dir(kwargs['input_path']):
        raise PathError(
            "Provided path for input directory does not exist!"
        )
    # --- Output Path Check ---
    if not Path.is_dir(kwargs['output_path']):
        if kwargs['force_dir']:
            warnings.warn(
                "Not able to find output directory; forcing creation!", 
                category = DirWarning)
            os.makedirs(Path(kwargs['output_path']), exist_ok=True)
        else:
            raise PathError(
                "Provided path for outnput directory does not exist!"
            )
    # --- Lists Length Check ---
    criteria = (len(kwargs['y1']) == len(kwargs['y2']) and
                len(kwargs['x1']) == len(kwargs['x2']) and
                len(kwargs['y2']) == len(kwargs['x1']) and
                len(kwargs['y1']) == len(kwargs['x2']) and
                len(kwargs['y1']) == len(kwargs['names_ls']))
    if not criteria:
        raise ValueError(
            "Annotation lists and names list must be of the same length!")


def _read_img_data(*args, **kwargs) -> DoubleIntTuple:
    # --- Reading Image ---
    img_sample = cv2.imread(Path(args[0], args[1]))
    # --- Reading Image Info ---
    si_height, si_width = img_sample.shape[:2] #> si: Sample Image
    # --- Return ---
    return img_sample, si_height, si_width


def _resize_calculation(*args, **kwargs) -> tuple:
    # --- Calculating Resize Ratio ---
    r = np.min([kwargs['height']/kwargs['si_height'], 
                kwargs['width']/kwargs['si_width']])
    # --- Resizing Calculation ---
    res_height = int(r*kwargs['si_height']) #> res: Resized
    res_width = int(r*kwargs['si_width'])
    # --- Calculating Padding Size ---
    y_padding = kwargs['height'] - res_height
    x_padding = kwargs['width'] - res_width
    # --- Return ---
    return res_height, res_width, y_padding, x_padding


def _img_resizer(*args, **kwargs):
    #> img_sample, res_width, res_height
    # --- Resizing ---
    resized_img = cv2.resize(
        args[0],
        (args[1], args[2]),
        interpolation = cv2.INTER_CUBIC)
    # --- Return ---
    return resized_img


def _img_padder(*args, **kwargs):
    # --- Value ---
    border_const_dict = {
        "Constant": cv2.BORDER_CONSTANT,
        "Reflect": cv2.BORDER_REFLECT,
        "Reflect_101": cv2.BORDER_REFLECT_101, 
        "Replicate": cv2.BORDER_REPLICATE}
    # --- Center Padding ---
    if kwargs['pbt'] == "Constant":
        padded_img = cv2.copyMakeBorder(
            kwargs['img_sample'],
            int(kwargs['y_padding']/2), #> Top
            int(kwargs['y_padding']/2), #> Bottom
            int(kwargs['x_padding']/2), #> Left
            int(kwargs['x_padding']/2), #> Right
            border_const_dict[kwargs['pbt']],
            value = kwargs['const_pbt_value'])
    else:
        padded_img = cv2.copyMakeBorder(
            kwargs['img_sample'],
            int(kwargs['y_padding']/2), #> Top
            int(kwargs['y_padding']/2), #> Bottom
            int(kwargs['x_padding']/2), #> Left
            int(kwargs['x_padding']/2), #> Right
            border_const_dict[kwargs['pbt']])
    # --- Return ---
    return padded_img


def _processed_img_save(*args, **kwargs) -> None:
    #> padded_img, output_path, img_name
    os.makedirs(kwargs['output_path'], exist_ok=True)
    # --- Creating Full Path ---
    save_path = os.path.join(
        kwargs['output_path'], kwargs['img_name'])
    # --- Saving ---
    cv2.imwrite(save_path, kwargs['padded_img'])


def _annotation_data_adjust(*args, **kwargs) -> tuple:
    #> si_height, si_width, res_height, res_width, 
    # y1i, y2i, x1i, x2i, y_padding, x_padding
    # --- Constant Values Creation ---
    height_r = kwargs['res_height']/kwargs['si_height']
    width_r = kwargs['res_width']/kwargs['si_width']
    height_padding = int(kwargs['y_padding']/2)
    width_padding = int(kwargs['x_padding']/2)
    # --- Adjusting Annotation Data ---
    kwargs['y1i'] = np.array(kwargs['y1i']*height_r, dtype=int)
    kwargs['y2i'] = np.array(kwargs['y2i']*height_r, dtype=int)
    kwargs['x1i'] = np.array(kwargs['x1i']*width_r, dtype=int)
    kwargs['x2i'] = np.array(kwargs['x2i']*width_r, dtype=int)
    # --- Return ---
    return (
        kwargs['y1i'] + height_padding, 
        kwargs['y2i'] + height_padding, 
        kwargs['x1i'] + width_padding, 
        kwargs['x2i'] + width_padding)


def _annotation_list_update(*args, **kwargs) -> tuple:
    #> y1, y2, x1, x2,
    #> y1i, y2i, x1i, x2i,
    #> count
    # --- Replacing Values ---
    kwargs['y1'][kwargs['count']] = kwargs['y1i']
    kwargs['y2'][kwargs['count']] = kwargs['y2i']
    kwargs['x1'][kwargs['count']] = kwargs['x1i']
    kwargs['x2'][kwargs['count']] = kwargs['x2i']
    # --- Return ---
    return kwargs['y1'], kwargs['y2'], kwargs['x1'], kwargs['x2']