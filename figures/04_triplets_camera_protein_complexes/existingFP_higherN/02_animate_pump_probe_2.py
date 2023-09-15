import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gs
from pathlib import Path
import numpy as np
import pandas as pd

# This script plots the output of the script 00_triplets_pump_probe...
# so it should only be run after the outputs of that script have been
# generated. It also imports .png exports from illustrator as part of an
# animation, so those files should have been generated (or not moved) as
# well.

# This script makes a graph of the output and also generates a teaching
# animation about pump probe pulse schemes.

# The simulation script splits each sample into multiple fluorophores
# objects ('replicates'). These replicates can be used to estimate the
# noise in the measurement. They can also be used to artificially create
# mixtures of different sizes (by summing the counts from replicates
# with different tumbling times). We do that trick here to generate a
# simulated curve for a 50-50 mixture of particules with tumbling time
# 450 and particles with tumbling time 3000.

# path setup & parameter configuration
cwd = Path(__file__).parents[0]
out_dir = cwd.parents[3] / 'tumbling_temp' / 'camera_existingFP_higherN'
out_dir.mkdir(exist_ok=True, parents=True)
plt.style.use(cwd.parents[1] / 'default.mplstyle')

diff2color = {'450': plt.cm.tab10(0), '900': plt.cm.tab10(1),
              '3000': plt.cm.tab10(3), '450-3000': plt.cm.tab10(4)}
# read in simulation output & process replicates to make mixtures
results = pd.read_csv('01_pump_probe_n1p0e+06_tqy0p0_9reps_2pulses.csv')
mixed_list = []
diff_list = [[(450, 450), (450, 450), (450, 450)], # sum("mix") tuples
             [(900, 900), (900, 900), (900, 900)], # avg list entries (replicates)
             [(3000, 3000), (3000, 3000), (3000, 3000)],
             [(450, 3000), (450, 3000), (450, 3000)]]
replicate_list = [[(0, 1), (2, 3), (4, 5)], # inner tuples will be summed
                  [(0, 1), (2, 3), (4, 5)], # list entries will be avg'd later
                  [(0, 1), (2, 3), (4, 5)],
                  [(6, 6), (7, 7), (8, 8)]]
label_list = ['450', '900', '3000', '450-3000']
for ds, rs, label in zip(diff_list, replicate_list, label_list):
    for entry, (d_tuple, r_tuple) in enumerate(zip(ds, rs)):
        d1, r1, d2, r2 = d_tuple[0], r_tuple[0], d_tuple[1], r_tuple[1]
        print('Mixing dtime %d rep %d with dtime %d rep %d' % (d1, r1, d2, r2))
        set1 = results.loc[(results['replicate'] == r1) &
                           (results['diff_time_ns'] == d1)]
        set2 = results.loc[(results['replicate'] == r2) &
                           (results['diff_time_ns'] == d2)]
        combo = pd.concat([set1, set2], ignore_index=True)
        gb = combo.groupby('delay_time_ns')
        x_sum = gb['x'].sum()
        y_sum = gb['y'].sum()
        assert np.all(set1['delay_time_ns'].values == set2['delay_time_ns'].values)
        df = pd.DataFrame({'x': x_sum.values,
                           'y': y_sum.values,
                           'delay_time_ns': gb.groups.keys(),
                           'diff_time_ns': label,
                           'entry': entry},
                          index=range(len(x_sum)))
        mixed_list.append(df)
mixed = pd.concat(mixed_list, ignore_index=True)
mixed['polarization'] = (mixed['x']-mixed['y']) / (mixed['x'] + mixed['y'])
mixed.to_csv('03_simulated_mixtures.csv')
# read in background images
im_list = []
for i in range(4):
    im = mpimg.imread(cwd.parents[0] / 'animation_stills' / ('%03d.png' % (i+1)))
    im_list.append(im)
im = np.array(im_list)

print('Animating', end='')
for i in range(120):
    print('.', end='')
    # plot the simulation output
    fig1 = plt.figure(figsize=(6, 5.5))
    grid = gs.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1.2, 1])
    ax1 = plt.subplot(grid[0, :]) # animation and graphic
    ax2 = plt.subplot(grid[1, 0]) # plot of the x counts
    ax3 = plt.subplot(grid[1, 1], sharex=ax2, sharey=ax2) # plot of the y counts
    ax4 = plt.subplot(grid[1, 2]) # plot of the y counts
    grid.update(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    ax1.axis('off')
    ax2.set_position([0.065, 0.07, 1.55/6, 1.55/5.5])
    ax3.set_position([0.405, 0.07, 1.55/6, 1.55/5.5])
    ax4.set_position([0.74, 0.07, 1.55/6, 1.55/5.5])
    # subpanel labels
    ax1.text(5, 40, 'A', fontsize=12, weight='bold', clip_on=False)
    ax1.text(5, 860, 'B', fontsize=12, weight='bold', clip_on=False)
    # plots of the simulation results
    for name, group in mixed.groupby('diff_time_ns'):
        x_means = group.groupby('delay_time_ns')['x'].mean()
        x_stds = group.groupby('delay_time_ns')['x'].std()
        delays = group.groupby('delay_time_ns').groups.keys()
        ax2.semilogx(delays, x_means, '.-',
                     label=name, color=diff2color[name], linewidth=2,
                     markersize=6)
        ax2.fill_between(delays,
                         x_means - x_stds,
                         x_means + x_stds,
                         color=diff2color[name], alpha=0.2)
        y_means = group.groupby('delay_time_ns')['y'].mean()
        y_stds = group.groupby('delay_time_ns')['y'].std()
        ax3.semilogx(delays, y_means, '.-',
                     label=name, color=diff2color[name], linewidth=2,
                     markersize=6)
        ax3.fill_between(delays,
                         y_means - y_stds,
                         y_means + y_stds,
                         color=diff2color[name], alpha=0.2)
        pol_means = group.groupby('delay_time_ns')['polarization'].mean()
        pol_stds = group.groupby('delay_time_ns')['polarization'].std()
        ax4.semilogx(delays, pol_means, '.-',
                     label=name, color=diff2color[name], linewidth=2,
                     markersize=6)
        ax4.fill_between(delays,
                         pol_means - pol_stds,
                         pol_means + pol_stds,
                         color=diff2color[name], alpha=0.2)
    handles, labels = ax2.get_legend_handles_labels() # a bit hacky
    order = [1, 3, 0, 2]
    ax2.legend([handles[idx] for idx in order],
               [labels[idx] for idx in order],
               loc='lower center', title='Tumbling Time (ns)', ncol=2,
               columnspacing=1.2, handlelength=1.5)
    for ax in [ax2, ax3, ax4]:
        ax.set_xlabel('Pump-Probe Delay (ns)', labelpad=1)
        ax.tick_params(axis='both', which='major', width=1)
        ax.spines[:].set_linewidth(1)
    ax2.set_ylabel(r'X Counts ($\parallel$ to probe)', labelpad=1)
    ax3.set_ylabel(r'Y Counts ($\perp$ to probe)', labelpad=1)
    ax4.set_ylabel(r'Polarization, (I$_X$ - I$_Y$) / (I$_X$ + I$_Y$)', labelpad=1)
    ax2.ticklabel_format(axis="y", style='sci', scilimits=(0,0))
    ax4.set_yticks(np.linspace(0, 0.5, 6))
    ax4.set_ylim(0, 0.5)
    ax2.set_yticks(np.linspace(0.5e3, 2.0e3, 4))
    ax2.set_ylim(0.5e3, 2.35e3)

    # indicate which simulation this is
    n = results['n_fluorophores'].values[0]*2 # should be the same for all
    tqy = results['triplet_QY'].values[0] # should be the same for all
    ax3.set_title(
        r'{:01.1e} fluorophores, $\Phi_{{triplet}}$ {:0.2f}'.format(n, tqy))

    # choose which background image to show
    if 32 < i < 36: # pump excitation hits sample
        ax1.imshow(im[1, :, :, :])
    elif 36 <= i < 91: # delay between pump and probe
        ax1.imshow(im[2, :, :, :])
    elif 91 <= i < 94: # probe excitation hits sample
        ax1.imshow(im[1, :, :, :])
    elif 117 <= i: # light reaches detector
        ax1.imshow(im[3, :, :, :])
    else:
        ax1.imshow(im[0, :, :, :])

    # draw the shutter in the correct location
    if i < 49: # pump sequence
        loc = (520, 420)
    elif 49 <= i < 56: # opening
        x = 520 - (i-49)*10
        loc = (x, 420)
    elif i >= 56:
        loc = (450, 420)
    shutter = plt.Rectangle(loc, 72, 10, facecolor='#000000')
    ax1.add_artist(shutter)

    # pump pulse
    if i < 9:
        y = i*14
        pump = plt.Rectangle((285, 204+y), 30, 30, facecolor='#00aeef')
        ax1.add_artist(pump)
    elif i < 27:
        x = (i-9)*14
        pump = plt.Rectangle((285 + x, 330), 30, 30, facecolor='#00aeef')
        ax1.add_artist(pump)
    elif i < 33:
        y = (i-27)*14
        pump = plt.Rectangle((541, 330-y), 30, 30, facecolor='#00aeef')
        ax1.add_artist(pump)
    elif i < 36:
        obj = plt.Rectangle((541, 125), 30 , 32, facecolor='#00aeef')
        ax1.add_artist(obj)
    # pump emission
    elif i < 46:
        y = (i-36)*14
        pump = plt.Rectangle((541, 260+y), 30, 30, facecolor='#23e000')
        ax1.add_artist(pump)
    elif i == 46: # first frame of hitting the shutter
        pump = plt.Rectangle((536, 385), 40, 35, facecolor='#23e000', alpha=0.5)
        ax1.add_artist(pump)
    elif i == 47: # second frame of hitting the shutter
        pump = plt.Rectangle((526, 380), 60, 40, facecolor='#23e000', alpha=0.2)
        ax1.add_artist(pump)
        
    # probe excitation
    if 56 <= i < 85:
        x = (i-56)*14
        pump = plt.Rectangle((120 + x, 330), 30, 30, facecolor='#ec008c')
        ax1.add_artist(pump)
    elif 85 <= i < 91:
        y = (i-85)*14
        pump = plt.Rectangle((541, 330-y), 30, 30, facecolor='#ec008c')
        ax1.add_artist(pump)
    elif 91 <= i < 94:
        obj = plt.Rectangle((541, 125), 30 , 32, facecolor='#ec008c')
        ax1.add_artist(obj)
    # probe emission
    elif 94 <= i < 109:
        y = (i-94)*14
        pump = plt.Rectangle((541, 260+y), 30, 30, facecolor='#23e000')
        ax1.add_artist(pump)
    elif 109 <= i < 117: # points headed to both detectors
        o = (i-109)*14
        pump_perp = plt.Rectangle((541, 470+o), 30, 30, facecolor='#23e000')
        ax1.add_artist(pump_perp)
        pump_para = plt.Rectangle((541-o, 470), 30, 30, facecolor='#23e000')
        ax1.add_artist(pump_para)
    elif 117 <= i < 120: # stationary at the detector
        pump_perp = plt.Rectangle((541, 574), 30, 30, facecolor='#23e000')
        ax1.add_artist(pump_perp)
        pump_para = plt.Rectangle((432, 470), 30, 30, facecolor='#23e000')
        ax1.add_artist(pump_para)  
    # save the output
    fig1.savefig(out_dir/('%04d.png'%i), dpi=300)
    plt.close()
