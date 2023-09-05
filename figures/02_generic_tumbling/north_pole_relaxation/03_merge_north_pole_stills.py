import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib as mpl
import numpy as np
from pathlib import Path
import pandas as pd

# This script combines the stills generated for different bead sizes in
# the script 001_generic_decay_cap and combines them into a series of
# stills with a nice graph and all four different bead sizes on it.

cur = Path(__file__).parents[0]
src = cur.parents[3] / 'tumbling_temp' / 'north_pole_relaxation' / 'individ'
out_path = cur.parents[3] / 'tumbling_temp' / 'north_pole_relaxation' / 'combined'
Path.mkdir(out_path, exist_ok=True)
mpl.rcParams['mathtext.default'] = 'regular'

# also import the quantification so we can provide a graph
path_csv = cur / '01_north_pole_relaxation.csv'
results = pd.read_csv(path_csv)
diff2diam = {21746: 40, 73394: 60, 339789: 100, 2718318: 200}
diff2color = {21746: '#0072b2', 73394: '#009e73', 339789: '#e69f00',
              2718318: '#cc79a7'}

for x in range(201): # number of frames that were previously generated - 201
    fname = '{:04d}.png'.format(x+1)
    print('.', end='')
    # set up a fresh figure
    plt.figure(figsize=(10,5.5))
    grid = gs.GridSpec(2, 4, width_ratios=[1.5, 1.5, 0.5, 2])
    grid.update(hspace=0, wspace=0)
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[0, 1])
    ax3 = plt.subplot(grid[1, 0])
    ax4 = plt.subplot(grid[1, 1])
    ax5 = plt.subplot(grid[:, 3])
    ax6 = plt.subplot(grid[:, 2]) # spacer 
    ax6.set_axis_off()
    ax_list = [ax1, ax2, ax3, ax4]
    for i, diff in enumerate(diff2diam.keys()):
        im = mpimg.imread(src / ('diff'+str(diff)+'_frame{:04d}.png'.format(x)))
        ax_list[i].imshow(im[150:1150, 225:1225, :])
        ax_list[i].set_axis_off()
    us_per_x = 0.2 # NOTE: HARDCODED TIME CONVERSION, should match 00 script
    ax1.text(750, -5, r'Time ($\mu$s): {:04.1f}'.format(x*us_per_x),
             ha='left', va='bottom')
    sx, sy = 500, 80
    ax1.text(sx, sy, '40 nm', ha='center', va='top')
    ax2.text(sx, sy, '60 nm', ha='center', va='top')
    ax3.text(sx, sy, '100 nm', ha='center', va='top')
    ax4.text(sx, sy, '200 nm', ha='center', va='top')
    ax3.text(680, 1150, 'Particle Orientations', ha='left', va='bottom',
             clip_on=False)
    # make the plot of the quantification in axis 5
    for name, group in results.groupby('diff_time_ns'):
        ax5.plot(group['time_ns'] / 1000,
                 group['avg_z_squared'], label=diff2diam[name],
                 linewidth=1, color=diff2color[name])
    ax5.axvline(x*us_per_x, color='#808080')
    ax5.set_ylabel('Alignment')
    ax5.set_ylim(0.3, 1.03)
    ax5.set_xlabel(r'Time ($\mu$s)')
    ax5.legend(loc=(0.64, 0.4), title='Diam. (nm)')
    plt.savefig(out_path / fname, bbox_inches='tight')
    plt.close()
