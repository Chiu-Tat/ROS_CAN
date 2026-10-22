"""Shared to-scale magnet symbols for the main UI and startup selector."""
import math
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.transforms import Affine2D

MAGNET_LENGTH_MM = 10.0
MAGNET_DIAMETER_MM = 4.0


def draw_magnet(ax, x, y, heading, free=False, directed=True, alpha=1.):
    """Planar cylinder symbol with a true 10 x 4 mm outer footprint."""
    length,diameter = MAGNET_LENGTH_MM,MAGNET_DIAMETER_MM
    cap = diameter/4
    half_body = (length-cap)/2
    transform = Affine2D().rotate(heading).translate(x,y)+ax.transData
    color,light,dark = (('#2c8b61','#78bf97','#18513b') if free else
                        ('#b85363','#dc929e','#72333f'))
    patches = (
            Ellipse((-half_body,0),cap,diameter,facecolor=dark,edgecolor=dark),
            Rectangle((-half_body,-diameter/2),2*half_body,diameter,facecolor=color,edgecolor=dark),
            Rectangle((-half_body,diameter*.12),2*half_body,diameter*.18,facecolor=light,edgecolor='none'),
            Ellipse((half_body,0),cap,diameter,facecolor=light,edgecolor=dark)) if free else (
            Ellipse((-half_body,0),cap,diameter,facecolor='#3d6faa',edgecolor='#294665'),
            Rectangle((-half_body,-diameter/2),half_body,diameter,facecolor='#3d6faa',edgecolor='#294665'),
            Rectangle((0,-diameter/2),half_body,diameter,facecolor='#c65363',edgecolor='#713441'),
            Rectangle((-half_body,diameter*.12),half_body,diameter*.18,facecolor='#81a8d1',edgecolor='none'),
            Rectangle((0,diameter*.12),half_body,diameter*.18,facecolor='#e2919d',edgecolor='none'),
            Ellipse((half_body,0),cap,diameter,facecolor='#e2919d',edgecolor='#713441'))
    for patch in patches:
        patch.set_transform(transform);patch.set_alpha(alpha);patch.set_zorder(4)
        patch.set_linewidth(.8);ax.add_patch(patch)
    if free and directed:
        direction = np.array([math.cos(heading),math.sin(heading)])
        ax.annotate('',xy=np.array([x,y])+direction*3,
            xytext=np.array([x,y])-direction*2,
            arrowprops={'arrowstyle':'->','color':'white','lw':1,'alpha':alpha},zorder=5)

