import numpy as np
import random
import torch

def _generate_masks(xs, ys):
    masks = []
    zeros = np.zeros([sum(xs),sum(ys)], dtype=bool)
    y_offset = 0
    for y in ys:
        x_offset = 0
        for x in xs:
            mask = zeros.copy()
            mask[y_offset:y_offset+y, x_offset:x_offset+x] = 1
            masks.append(mask)
            x_offset += x
        y_offset += y
    return masks

def division_masks_from_spec(specs):
    wides = {k: _generate_masks(**v) for k, v in specs.items()}
    ret = {k: [masks, [np.rot90(m).copy() for m in masks]] for k, masks in wides.items()}
    return ret

DIVISION_SPECS_14_14 = {
    1: {"xs": [14], "ys": [14]},
    2: {"xs": [14], "ys": [7, 7]},
    4: {"xs": [7, 7], "ys": [7, 7]},
    8: {"xs": [7, 7], "ys": [4, 3, 3, 4]},
    16: {"xs": [4, 3, 3, 4], "ys": [4, 3, 3, 4]},
    3: {"xs": [14], "ys": [5, 4, 5]},
    6: {"xs": [7, 7], "ys": [5, 4, 5]},
    9: {"xs": [5, 4, 5], "ys": [5, 4, 5]},
    12: {"xs": [4, 3, 3, 4], "ys": [5, 4, 5]},
}

DIVISION_SPECS_12_12 = {
    1: {"xs": [12], "ys": [12]},
    2: {"xs": [12], "ys": [6, 6]},
    4: {"xs": [6, 6], "ys": [6, 6]},
    8: {"xs": [6, 6], "ys": [3, 3, 3, 3]},
    16: {"xs": [3, 3, 3, 3], "ys": [3, 3, 3, 3]},
    3: {"xs": [12], "ys": [4, 4, 4]},
    6: {"xs": [6, 6], "ys": [4, 4, 4]},
    9: {"xs": [4, 4, 4], "ys": [4, 4, 4]},
    12: {"xs": [3, 3, 3, 3], "ys": [4, 4, 4]},
}

DIVISION_MASKS = {
    12: division_masks_from_spec(DIVISION_SPECS_12_12),
    14: division_masks_from_spec(DIVISION_SPECS_14_14)
}
