"""Color/metric geometry checks for the green cylindrical free magnet."""
import math

import numpy as np

from .core import calibrated_line

DETECTOR = 'green_cylinder_v1'


def positive_y_heading(axis, boundary_margin=0.0):
    """Resolve an undirected EM-frame line using N in the open +y half-plane."""
    if not math.isfinite(axis) or not 0 <= boundary_margin < math.pi/2:
        raise ValueError('Invalid initial line angle or boundary margin')
    heading = float(axis) % math.pi
    if heading <= boundary_margin or heading >= math.pi-boundary_margin:
        raise ValueError('Initial line is too close to horizontal; point N clearly into the +y half-plane')
    return heading


def green_candidate(blob, affine, settings):
    """Return (world center, undirected world axis, metric size) or reject.

    The camera supplies a tight rotated rectangle around the green silhouette.
    Transform its edges, rather than its axis-aligned pixel box: the calibration
    may include rotation, reflection, unequal scale, and shear.
    """
    if blob.get('kind') != DETECTOR:
        raise ValueError('Expected green cylinder detection; upload current camera code')
    xy, axis = calibrated_line(blob, affine)
    corners = np.asarray(blob['corners_px'], dtype=float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError('Invalid green silhouette corners')
    edges = np.roll(corners, -1, axis=0)-corners
    lengths = np.linalg.norm(edges @ np.asarray(affine, float)[:, :2].T, axis=1)
    opposite = np.array([(lengths[0]+lengths[2])/2, (lengths[1]+lengths[3])/2])
    width, length = sorted(opposite)
    pixels = float(blob['pixels'])
    roundness = float(blob['roundness'])
    if not math.isfinite(pixels) or not math.isfinite(roundness):
        raise ValueError('Nonfinite green silhouette quality')
    if pixels < settings['min_green_pixels'] or not 0 <= roundness <= settings['max_roundness']:
        raise ValueError('Too few green pixels or silhouette too round')
    if not settings['length_range_m'][0] <= length <= settings['length_range_m'][1]:
        raise ValueError('Green object length does not match the 11 mm magnet')
    if not settings['diameter_range_m'][0] <= width <= settings['diameter_range_m'][1]:
        raise ValueError('Green object width does not match the 4 mm magnet')
    if not settings['aspect_range'][0] <= length/width <= settings['aspect_range'][1]:
        raise ValueError('Green object aspect ratio does not match the magnet')
    # A degenerate/nonrectangular packet must not pass by averaging edge lengths.
    if np.min(lengths) <= 0 or abs(np.cross(edges[0], edges[1])) < 1e-9:
        raise ValueError('Degenerate green silhouette')
    # Moment axis should agree with the long silhouette edge, modulo pi.
    long_edge = edges[int(np.argmax(lengths))] @ np.asarray(affine, float)[:, :2].T
    edge_axis = math.atan2(long_edge[1], long_edge[0])
    disagreement = abs((edge_axis-axis+math.pi/2) % math.pi-math.pi/2)
    if disagreement > settings['max_axis_box_disagreement_rad']:
        raise ValueError('Silhouette axis and long edge disagree')
    return xy, axis, {'length_m': float(length), 'diameter_m': float(width),
                      'green_pixels': int(pixels)}
