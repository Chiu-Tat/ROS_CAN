#!/usr/bin/env python3
"""Fit pixel-to-coil-world affine calibration from measured reference points."""
import argparse
import json
from pathlib import Path

import numpy as np


def fit(points):
    data = np.asarray(points, float)
    if data.ndim != 2 or data.shape[0] < 4 or data.shape[1] != 4 or not np.isfinite(data).all():
        raise ValueError('Supply >=4 [pixel_x,pixel_y,world_x_m,world_y_m] points')
    design = np.column_stack([data[:, :2], np.ones(len(data))])
    coefficients, _, rank, _ = np.linalg.lstsq(design, data[:, 2:], rcond=None)
    if rank != 3 or abs(np.linalg.det(coefficients[:2])) < 1e-15:
        raise ValueError('Calibration points must span a 2-D region')
    errors = np.linalg.norm(design @ coefficients-data[:, 2:], axis=1)
    return {'affine':coefficients.T.tolist(), 'frame_id':'emns_world',
            'rms_error_m':float(np.sqrt(np.mean(errors**2))),
            'max_error_m':float(errors.max()), 'points':data.tolist()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('points', help='JSON list of measured reference points')
    parser.add_argument('output', help='New output file (existing files are preserved)')
    args = parser.parse_args()
    result = fit(json.loads(Path(args.points).read_text()))
    with open(args.output, 'x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('Calibration RMS %.3f mm, max %.3f mm' % (1000*result['rms_error_m'],1000*result['max_error_m']))
