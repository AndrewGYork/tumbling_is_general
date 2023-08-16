import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.image as mpimg
from matplotlib import patches
import numpy as np
import pandas as pd
from pathlib import Path

## Run this script AFTER the 01_simulate_photoswitching_power script to
## plot the output from that script.

## This script also incorporates some Illustrator-generated static
## graphics describing the analytes and optics for this approach.

## It also includes a dots-on-the-sphere animation of orientation for
## two sizes at a given saturation to help build intuition about why the
## power-variation approach works. These stills are generated within the
## simulation script.

# import the simulation output for plotting & some setup
cwd = Path(__file__).parents[0]
adir = cwd.parents[3] / 'tumbling_temp' / 'photoswitching' / 'dots_on_a_sphere'
out_path = cwd.parents[3] / 'tumbling_temp' / 'photoswitching' / 'sequence'
out_path.mkdir(exist_ok=True, parents=True)
plt.style.use(cwd.parents[1] / 'default.mplstyle')
results = pd.read_csv('photoswitching_summed_counts_1e2.csv')
cdfs = pd.read_csv('photoswitching_cdfs_1e2.csv')
diff2label = {750: '750: monomer', 1200: '1200: small',
              3000: '3000: medium', 6000: '6000: large'}
diff2color = {750: plt.cm.tab10(0), 1200: plt.cm.tab10(1),
              3000: plt.cm.tab10(5), 6000: plt.cm.tab10(3)}

# import the static images
pdir = cwd.parents[0]
analyte_small = mpimg.imread(pdir/'analytes_monomer.png')
analyte_large = mpimg.imread(pdir/'analytes_large_aggregate.png')
optics = mpimg.imread('photoswitching_optics.png')

# Pre-calculate the y/x ratios to make plotting faster. Also
# pre-calculate the coordinates we will circle and show animations of
y = results['counts_y']; x = results['counts_x']
results['polarization'] = (y - x) / (y + x)
s1 = results.loc[(results['diffusion_time_ns'] == 750) &
                 (results['saturation'] == 1.25)]
y1 = s1['polarization'].mean()
s2 = results.loc[(results['diffusion_time_ns'] == 6000) &
                 (results['saturation'] == 1.25)]
y2 = s2['polarization'].mean()

for i in range(210): # 100 us every 0.5 us, with 10 static frames at the end - 210
    print('.', end='')
    j = i if i < 200 else 199 # other index allows for pausing at the end
    # positioning of axes
    fig = plt.figure(figsize=(6, 4))
    grid = gs.GridSpec(8, 3, width_ratios=[2.5, 1.6, 1],
                       height_ratios=[0.5, 1, 1, 0.5, 1, 1, 0.5, 0.5])
    ax1 = plt.subplot(grid[:2, 0]) # optics
    ax2 = plt.subplot(grid[2:, 0]) # graph
    ax3 = plt.subplot(grid[1:4, 1]) # dots on a sphere 1
    ax4 = plt.subplot(grid[4:7, 1]) # dots on a sphere 2
    ax5 = plt.subplot(grid[7:, 1]) # legend for dots on a sphere
    ax6 = plt.subplot(grid[1:2, 2]) # cartoon for dots 1
    ax7 = plt.subplot(grid[2:3, 2]) # bar chart for dots 1
    ax8 = plt.subplot(grid[4:5, 2]) # cartoon for dots 2
    ax9 = plt.subplot(grid[5:6, 2]) # bar chart for dots 2
    grid.update(left=0.02, right=1, bottom=0, top=1, hspace=0, wspace=0)
    ax1.set_position([0.08, 0.7, 2.1/6, 1.05/4])
    ax2.set_position([0.09, 0.13, 2.25/6, 2.25/4])
    for ax in [ax1, ax3, ax4, ax5, ax6, ax8]:
        ax.set_axis_off()
    # subpanel lettering
    ax1.text(-100, 0, 'A', fontsize=12, weight='bold')
    ax1.text(-100, 300, 'B', fontsize=12, weight='bold')
    ax1.text(900, 0, 'C', fontsize=12, weight='bold')
    ax1.text(50, -25, 'Photoswitching, 1e2 Fluorophores', fontsize=8)
    # import the images for two different sizes
    for ax, diff in zip([ax3, ax4], ['7p5e+02', '6p0e+03']):
        fstem = 'photoswitching_dt'+diff +'_sat1p250_100_frame{:04d}'.format(j+1)
        im = mpimg.imread(adir/(fstem+'.png'))
        ax.imshow(im[50:450, 50:450, :])
    c750 = cdfs.loc[cdfs['diffusion_time_ns'] == 750]
    c6000 = cdfs.loc[cdfs['diffusion_time_ns'] == 6000]
    ax3.text(200, 10, 'Monomer', color='#1f77b4', fontsize=10,
             ha='center')
    ax4.text(200, 10, 'Large Aggregate', color='#d62728', fontsize=10,
             ha='center')
    # add the time annotation
    time_us = c750['time_ns'].values[j] / 1000 # assuming 750, 6000 match
    ax3.text(275, -20, r'Time ($\mu$s): {:04.1f}'.format(time_us),
             ha='left', va='bottom', fontsize=10)
    # generate a legend for the dots on a sphere states
    dot1 = plt.Circle((0.2, 0.7), 0.07, color='#808080')
    dot2 = plt.Circle((0.2, 0.25), 0.07, color='#23e000')
    ax5.add_artist(dot1); ax5.add_artist(dot2)
    ax5.set_aspect('equal')
    ax5.set_ylim(0,1); ax5.set_xlim(0,5)
    ax5.text(0.4, 0.7, 'ground state', va='center')
    ax5.text(0.4, 0.25, 'excited state', va='center')
    # legend for the emission component of dots on a sphere
    rect1 = plt.Rectangle((2.95, 0.7), 0.2, 0.06, color='#be03fd', alpha=0.3)
    rect2 = plt.Rectangle((3, 0.20), 0.06, 0.2, color='#ffa500', alpha=0.3)
    ax5.add_artist(rect1); ax5.add_artist(rect2)
    ax5.text(3.3, 0.7, 'count in X', va='center')
    ax5.text(3.3, 0.25, 'count in Y', va='center')
    # display the cartoons for the analytes and the optics
    ax1.imshow(optics)
    ax6.imshow(analyte_small)
    ax8.imshow(analyte_large)
    # display the plot of the photobleaching results
    for name, group in results.groupby('diffusion_time_ns'):
        # get an average and sd across the replicates
        ratio_mean = group.groupby('saturation')['polarization'].mean()
        ratio_std = group.groupby('saturation')['polarization'].std()
        saturations = group.groupby('saturation').groups.keys()
        ax2.semilogx(saturations, ratio_mean,
                     '.-', label=diff2label[name],
                     linewidth=1.5, markersize=6,
                     color=diff2color[name])
        ax2.fill_between(saturations,
                         ratio_mean - ratio_std,
                         ratio_mean + ratio_std,
                         color=diff2color[name], alpha=0.3)
    ax2.legend(title='Tumbling Time (ns)', fontsize=8,
               handlelength=1.5)
    ax2.set_xlabel('Off-Switching Intensity', fontsize=10)
    ax2.set_ylabel('Polarization, (Y-X) / (Y+X)', fontsize=10)
    ax2.set_ylim(-0.05, 0.55)
    ax2.set_yticks(np.linspace(0, 0.5, 6))
    ax2.tick_params(axis='both', which='major', width=1, labelsize=8)
    ax2.spines[:].set_linewidth(1)
    # circle the points we are measuring & draw arrows to them
    ax2.plot(1.25, y1, marker='s', markeredgecolor='#1f77b4', markersize=8,
             markerfacecolor='none', markeredgewidth=1.5, alpha=0.5)
    ax2.plot(1.25, y2, marker='s', markeredgecolor='#d62728', markersize=8,
             markerfacecolor='none', markeredgewidth=1.5, alpha=0.5)
    xyA1 = [10, 200] # coordinates on the animation
    xyB1 = [1.7, y1+0.005] # coordinates on the plot
    arrow1 = patches.ConnectionPatch(xyA1, xyB1, coordsA=ax3.transData,
                                     coordsB=ax2.transData,
                                     color='#1f77b4', arrowstyle="-|>",
                                     mutation_scale=20, linewidth=1.5,
                                     alpha=0.5)
    fig.patches.append(arrow1)
    xyA2 = [10, 250] # coordinates on the animation
    xyB2 = [1.78, y2-0.005] # coordinates on the plot
    arrow2 = patches.ConnectionPatch(xyA2, xyB2, coordsA=ax4.transData,
                                     coordsB=ax2.transData,
                                     color='#d62728', arrowstyle="-|>",
                                     mutation_scale=20, linewidth=1.5,
                                     alpha=0.5)
    fig.patches.append(arrow2)
    # Make a bar chart that displays the counts as they are increasing
    # for the smaller sample
    ax7.bar(range(2), [c750['y_cdf'].values[-1], c750['x_cdf'].values[-1]],
            fill=None, edgecolor='#808080')
    ax7.bar(range(2), [c750['y_cdf'].values[j], c750['x_cdf'].values[j]],
            alpha=0.5, color='#1f77b4')
    ax7.set_xticks(range(2))
    ax7.set_xticklabels(['Y', 'X'])
    ax7.set_position([0.84, 0.61, 0.13, 0.25*2/3]) 
    ax7.ticklabel_format(axis="y", style='sci', scilimits=(0,0))
    ax7.set_ylim(0, 4e4)
    ax7.set_ylabel('Counts')
    ax7.text(0.5, -2e4, 'Y / X = {:0.2f}'.format(
        c750['y_cdf'].values[-1] / c750['x_cdf'].values[-1]), clip_on=False,
             ha='center')
    # Make a bar chart that displays the counts as they are increasing
    # for the larger sample
    ax9.bar(range(2), [c6000['y_cdf'].values[-1], c6000['x_cdf'].values[-1]],
            fill=None, edgecolor='#808080')
    ax9.bar(range(2), [c6000['y_cdf'].values[j], c6000['x_cdf'].values[j]],
            alpha=0.5, color='#d62728')
    ax9.set_xticks(range(2))
    ax9.set_xticklabels(['Y', 'X'])
    ax9.set_position([0.84, 0.15, 0.13, 0.25*2/3])
    ax9.ticklabel_format(axis="y", style='sci', scilimits=(0,0))
    ax9.set_ylim(0, 4e4)
    ax9.set_ylabel('Counts')
    ax9.text(0.5, -2e4, 'Y / X = {:0.2f}'.format(
        c6000['y_cdf'].values[-1] / c6000['x_cdf'].values[-1]), clip_on=False,
             ha='center')
    # save the frame for later gif rendering    
    plt.savefig(out_path / '{:04d}.png'.format(i), dpi=200)
    plt.close()
