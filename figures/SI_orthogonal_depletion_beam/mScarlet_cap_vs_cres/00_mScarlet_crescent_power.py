from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import Divider, Size

# This script plots the results of changing the crescent power on
# mScarlet tumbling.

# This script assumes that you have already run
# inspect_triplets_delay_series_8 and _16 to extract raw tumbling data.
# It also assumes that you have run the 00_triplet_plotting.py
# preprocessing script on each individual days' worth of data.

# Read in the triplet counts and polarization ratios (already g factor
# corrected) for each day.
current_dir = Path(__file__).parents[0]
pdir = current_dir.parents[3] / 'tumbling_temp'
p0419 = pdir / 'source_data' / '2023-04-19_mScarlet_crescent.sptw' / 'analysis'
df0419 = pd.read_csv(p0419 / '15_2023-04-19_CML_mScarlet_triggered_triplets.csv')
df0419['date'] = 20230419; df0419['version'] = 1
# Also import one additional replicate of crescent 2 data from 04-24
# from that tumbling run so that we have n=3 for each crescent power.
p0424 = pdir / 'source_data' / '2023-04-24_mScarlet_tumbling.sptw' / 'analysis'
df0424 = pd.read_csv(p0424 / '15_2023-04-24_mScarlet_triggered_triplets.csv')
df0424['date'] = 20230424; df0424['version'] = 1
drop_me = df0424.index[(df0424['bead_diam_nm'] != 100) |
                       (df0424['trigger_frame'] > 8)]
df0424.drop(drop_me, inplace=True)
p0426 = pdir / 'source_data' / '2023-04-26_mScarlet_tumbling.sptw' / 'analysis'
df0426 = pd.read_csv(p0426 / '15_2023-04-26_mScarlet_triggered_triplets.csv')
df0426['date'] = 20230426 # versions were already included in metadata
# We only want to plot the crescent power test data
drop_me = df0426.index[(df0426['bead_diam_nm'] != 100) |
                       (df0426['trigger_frame'] > 8)]
df0426.drop(drop_me, inplace=True)
df = pd.concat([df0419, df0424, df0426], ignore_index=True)

df['time_us'] = (df['frame'] - 1) * 60
plt.style.use(current_dir.parents[1] / 'default.mplstyle')
diam2color = {100: '#e69f00'}
power2color = {0: plt.cm.Wistia(0.1), 2: plt.cm.Wistia(0.5),
               4: plt.cm.Wistia(0.99)}

# Let's sanity check our parsing of the above data
for name, group in df.groupby('crescent_power'):
    assert len(group) == 21 # triplicate of 7 points

# first, let's plot the polarization ratios at each delay & crescent power
fig1 = plt.figure(figsize=(4,4), dpi=300)
h = [Size.Fixed(1.0), Size.Fixed(2)]
v = [Size.Fixed(0.7), Size.Fixed(2)]
divider = Divider(fig1, (0, 0, 1, 1), h, v, aspect=False)
axs1 = fig1.add_axes(divider.get_position(),
                     axes_locator=divider.new_locator(nx=1, ny=1))
for name, group in df.groupby('crescent_power'):
    ratio_means = group.groupby('time_us')['polarization'].mean()
    ratio_stds = group.groupby('time_us')['polarization'].std()
    delays = group.groupby('time_us').groups.keys()
    axs1.plot(delays, ratio_means, '.-', markersize=5, label=name,
              color=power2color[name])
    axs1.fill_between(delays, ratio_means - ratio_stds,
                      ratio_means + ratio_stds, alpha=0.2,
                      color=power2color[name])    
plt.title('mScarlet')
for ax in fig1.axes:
    ax.grid('on')
    ax.legend(title='Crescent Power (%)', fontsize=8, columnspacing=1.3,
              handlelength=1.3, title_fontsize=8, framealpha=1,
              handletextpad=0.7, borderpad=0.5, loc='upper right')
    ax.set_xlabel(r'Pump-Probe Delay Time ($\mu$s)')
    ax.set_ylabel('Polarization')
    ax.set_ylim(-0.30, 0)
plt.savefig('01_mScarlet_crescent_pol.pdf', bbox_inches='tight',
            transparent=True)

# now let's plot the change in total counts with crescent power so we
# can try to relate this to some sort of saturation unit in the
# simulations
fig2 = plt.figure(figsize=(4,4), dpi=300)
h = [Size.Fixed(1.0), Size.Fixed(1.7)]
v = [Size.Fixed(0.7), Size.Fixed(1.7)]
divider = Divider(fig2, (0, 0, 1, 1), h, v, aspect=False)
axs2 = fig2.add_axes(divider.get_position(),
                     axes_locator=divider.new_locator(nx=1, ny=1))
df['total_cts'] = df['A_bkgd_sub'] + df['B_bkgd_sub']
df.sort_values(by='crescent_power', inplace=True)
delay360 = df.loc[df['time_us'] == 360] # should be frame 7
max_cts = delay360.loc[delay360['crescent_power'] == 0, 'total_cts'].sum()
axs2.plot(delay360.crescent_power.unique(),
          delay360.groupby('crescent_power')['total_cts'].sum() / max_cts,
          '.-', color=diam2color[100], markersize=3)
plt.title('mScarlet, 100 nm')
for ax in fig2.axes:
    ax.grid('on')
    ax.set_xlabel('775 Intensity (%)')
    ax.set_ylabel(r'Norm. Counts at Delay 360 $\mu$s')
    ax.set_ylim(0.1, 1.05)
plt.savefig('02_mScarlet_crescent_cts360us.pdf', bbox_inches='tight',
            transparent=True)

plt.show()

