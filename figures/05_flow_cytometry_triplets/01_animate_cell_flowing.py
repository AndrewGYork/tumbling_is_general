import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gs
from pathlib import Path
import numpy as np
import pandas as pd

# This script plots the results of the flow cytometry simulation. It
# also animates a cell moving down a tube and incorporates an
# illustrator-generated cartoon of the different sample types. This
# script should not be run until the 00_simulate_flow_cytometry...
# script has executed.

# read in the background images
cwd = Path(__file__).parents[0]
out_dir = cwd.parents[2] / 'tumbling_temp' / 'flow_sequence'
out_dir.mkdir(exist_ok=True, parents=True)
im_list = []
for i in range(5):
    im = mpimg.imread(cwd / 'flow_stills' / ('%03d.png' % (i+1)))
    im_list.append(im)
im = np.array(im_list)
print(im.shape)
# read in the image with the proteins
proteins = mpimg.imread(cwd / 'complexes_for_flow.png')

# read in the data for plotting and the style sheet for formatting
results = pd.read_csv('flow_cytometry_circ_probe_12reps.csv')
x = results['counts_x']; y = results['counts_y']
results['polarization'] = (y - x) / (x + y)
plt.style.use(cwd.parents[0] / 'default.mplstyle')
diff2color = {450: plt.cm.tab10(0), 900: plt.cm.tab10(1),
              3000: plt.cm.tab10(3)}

for i, x in enumerate(range(0, 900, 5)):
    print('.', end='')
    fig = plt.figure(figsize=(8, 9.35))
    grid = gs.GridSpec(2, 2, width_ratios=[1, 1], height_ratios = [1, 1])
    ax1 = plt.subplot(grid[0, :])
    ax2 = plt.subplot(grid[1, 0])
    ax3 = plt.subplot(grid[1, 1])
    grid.update(left=0.0, right=1.0, bottom=0.1, top=1.0, wspace=0, hspace=0.1)
    state = 'glowing'
    if  325 < x < 395: # probe 2
        ax1.imshow(im[3, :, :, :], aspect='equal')
    elif  735 < x < 805: # probe 3
        ax1.imshow(im[4, :, :, :], aspect='equal')
    elif (248 < x < 318) or (555 < x < 625): # in one of the blue beams w/o detector
        ax1.imshow(im[0, :, :, :], aspect='equal')
    elif (25 < x < 60): # pump 1
        ax1.imshow(im[1, :, :, :], aspect='equal')
    elif (60 < x < 95): # probe 1
        ax1.imshow(im[2, :, :, :], aspect='equal')
    elif (318 <= x <= 325) or (625 <= x <= 735):
        state = 'triplet'
        ax1.imshow(im[0, :, :, :], aspect='equal')
    else: # in between delays
        state = 'none'
        ax1.imshow(im[0, :, :, :], aspect='equal')
    if state == 'glowing':
        fcolor = '#39ff14'; ecolor='#39ff14'; a=1
        halo = plt.Circle((x, 219), 20, facecolor=fcolor, edgecolor=fcolor,
                          alpha=0.3)
        ax1.add_artist(halo)
    elif state == 'triplet':
        fcolor = '#f02aff'; ecolor='#f02aff'; a=0.2
        halo = plt.Circle((x, 219), 20, facecolor=fcolor, edgecolor=fcolor,
                          alpha=0.1)
        ax1.add_artist(halo)        
    else:
        fcolor = '#a6a6a6'; ecolor='#808080'; a=1
    cell = plt.Circle((x, 219), 16, facecolor=fcolor, edgecolor=ecolor,
                      alpha=a)
    ax1.add_artist(cell)
    ax1.set_axis_off()

    # display the graph on one of the bottom axes
    for name, group in results.groupby('diffusion_time_ns'):
        ratio_means = group.groupby('delay_ns')['polarization'].mean()
        ratio_stds = group.groupby('delay_ns')['polarization'].std()
        delays = group.groupby('delay_ns').groups.keys()
        ax3.semilogx(delays, ratio_means,
                     '.-', label=round(name), linewidth=2,
                     markersize=8, color=diff2color[name])
        ax3.fill_between(delays,
                         ratio_means - ratio_stds,
                         ratio_means + ratio_stds,
                         color=diff2color[name], alpha=0.2)
    ax3.legend(title='Tumbling Time (ns)', loc='upper right',
               fontsize=12, title_fontsize=12)
    ax3.set_ylabel('Polarization, (Y-X) / (Y+X)', fontsize=16)
    ax3.set_xlabel('Probe Delay (ns)', fontsize=16)
    ax3.spines[:].set_linewidth(1.5)
    ax3.tick_params(axis='both', which='major', labelsize=14, width=1)
    # generate a rectangle to highlight what's being measured
    if 25 < x <= 95: # delay 1
        box = plt.Rectangle((185, 0.05), 30, 0.3, fill=False,
                            edgecolor='#ec008c')
        ax3.add_artist(box)
    elif 248 < x <= 395: # delay 2
        box = plt.Rectangle((750, -0.01), 100, 0.2, fill=False,
                            edgecolor='#ec008c')
        ax3.add_artist(box)
    elif 555 < x <= 805: # delay 3
        box = plt.Rectangle((3000, -0.01), 400, 0.04, fill=False,
                            edgecolor='#ec008c')
        ax3.add_artist(box)        

    # display the images of complexes on the other bottom axis
    ax2.set_position([0, 0.07, 0.4, 0.4*9.35/8])
    ax2.imshow(proteins, aspect='equal')
    ax2.set_axis_off()
    
    fig.savefig(out_dir/('%03d.png'%i), dpi=200)
    plt.close()
