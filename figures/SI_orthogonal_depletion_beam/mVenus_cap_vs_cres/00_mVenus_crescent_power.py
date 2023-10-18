from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import Divider, Size

# This script plots the results of changing the crescent power on
# mVenus tumbling.

# This script assumes that you have already run
# inspect_triplets_delay_series_8 to extract raw tumbling data.
# It also assumes that you have run the 00_triplet_plotting.py
# preprocessing script on each individual days' worth of data.

# Read in the triplet counts and polarization ratios (already g factor
# corrected) for each day.
current_dir = Path(__file__).parents[0]
pdir = current_dir.parents[3] / 'tumbling_temp'
p0413 = pdir / 'source_data' / '2023-04-13_Venus_tumbling.sptw' / 'analysis'
df0413 = pd.read_csv(p0413 / '15_2023-04-13_CML_mVenus_triggered_triplets.csv')
df0413['version'] = 1
drop_me = df0413.index[(df0413['bead_diam_nm'] != 100) |
                       (df0413['trigger_frame'] > 8)]
df0413.drop(drop_me, inplace=True)
# Let's get the 2023-05-01 data now
p0501 = pdir / 'source_data' / '2023-05-01_mVenus_crescent.sptw' / 'analysis'
df0501 = pd.read_csv(p0501 / '15_2023-05-01_mVenus_triggered_triplets.csv')
df_list = [df0413, df0501]
# Let's pull the 0.5% crescent data from other days' that were used in
# the main 15 point tumbling analyses
for day in ['2023-04-06', '2023-04-10', '2023-04-11']:
    path = pdir / 'source_data' / (day + '_Venus_tumbling.sptw') / 'analysis'
    df1 = pd.read_csv(path / ('15_%s_CML_mVenus_triggered_triplets.csv' % day))
    df1['version'] = 1
    drop_me = df1.index[(df1['bead_diam_nm'] != 100)|(df1['trigger_frame'] > 8)]
    df1.drop(drop_me, inplace=True)
    df_list.append(df1)
df = pd.concat(df_list, ignore_index=True)

df['time_us'] = (df['frame'] - 1) * 60
plt.style.use(current_dir.parents[1] / 'default.mplstyle')
diam2color = {100: '#e69f00'}
power2color = {0: plt.cm.Wistia(0.1), 0.5: plt.cm.Wistia(0.33),
               1: plt.cm.Wistia(0.66), 2: plt.cm.Wistia(0.99)}

# Let's sanity check our parsing of the above data
for name, group in df.groupby('crescent_power'):
    print(name, 'group size', len(group))

# first, let's plot the polarization ratios
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
plt.title('mVenus')
for ax in fig1.axes:
    ax.grid('on')
    ax.legend(title='Crescent Power (%)', fontsize=8, columnspacing=1.3,
              handlelength=1.3, title_fontsize=8, framealpha=1,
              handletextpad=0.7, borderpad=0.5, loc='upper right')
    ax.set_xlabel(r'Pump-Probe Delay Time ($\mu$s)')
    ax.set_ylabel('Polarization')
    ax.set_ylim(-0.3, 0)
plt.savefig('01_mVenus_crescent_pol.pdf', bbox_inches='tight',
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
# this is imperfect, but we'll adjust based on the total number of
# recordings since we have different # of replicates in each group
n = delay360.groupby('crescent_power')['filename'].count().values
n_rel = n[0] / n
max_cts = delay360.loc[delay360['crescent_power'] == 0, 'total_cts'].sum()
axs2.plot(delay360.crescent_power.unique(),
          delay360.groupby('crescent_power')['total_cts'].sum() * n_rel / max_cts,
          '.-', color=diam2color[100], markersize=3)
plt.title('mVenus, 100 nm')
for ax in fig2.axes:
    ax.grid('on')
    ax.set_xlabel('775 Intensity (%)')
    ax.set_ylabel(r'Norm. Counts at 360 $\mu$s')
    ax.set_ylim(0.1, 1.05)
plt.savefig('02_mVenus_crescent_cts360us.pdf', bbox_inches='tight',
            transparent=True)

plt.show()

