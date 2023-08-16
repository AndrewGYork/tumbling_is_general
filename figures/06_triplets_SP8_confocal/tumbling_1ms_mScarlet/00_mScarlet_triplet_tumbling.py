from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import Divider, Size

# This script plots how the polarization ratio and anisotropy change for
# the triggered fluorescence from mScarlet at increasing trigger delays.
# (In other words, it plots a tumbling curve.)

# This script assumes that you have already run
# inspect_triplets_delay_series_8 and _16 to extract raw tumbling data.
# It also assumes that you have run the triplet_anisotropy_plotting.py
# preprocessing script on each individual days' worth of data.

# Read in the triplet counts and polarization ratios (already g factor
# corrected) for each day.
current_dir = Path(__file__).parents[0]
pdir = current_dir.parents[3] / 'tumbling_temp'
p0420 = pdir / 'source_data' / '2023-04-20_mScarlet_tumbling.sptw' / 'analysis'
df0420 = pd.read_csv(p0420 / '15_2023-04-20_mScarlet_triggered_triplets.csv')
df0420['date'] = 20230420; df0420['version'] = 1
# I used a higher amount of protein on the 60 and 200 nm beads on 04-20.
# Don't include these in the final analysis.
drop60_200 = df0420.index[(df0420['bead_diam_nm'] == 60) |
                          (df0420['bead_diam_nm'] == 200)]
df0420.drop(drop60_200, inplace=True)
p0424 = pdir / 'source_data' / '2023-04-24_mScarlet_tumbling.sptw' / 'analysis'
df0424 = pd.read_csv(p0424 / '15_2023-04-24_mScarlet_triggered_triplets.csv')
df0424['date'] = 20230424; df0424['version'] = 1
p0426 = pdir / 'source_data' / '2023-04-26_mScarlet_tumbling.sptw' / 'analysis'
df0426 = pd.read_csv(p0426 / '15_2023-04-26_mScarlet_triggered_triplets.csv')
df0426['date'] = 20230426 # versions were already included in metadata
# I had some different crescent powers in the 0426 data for a different
# experiment. Don't use those.
drop_cres = df0426.index[df0426['crescent_power'] != 2]
df0426.drop(drop_cres, inplace=True)
df = pd.concat([df0420, df0424, df0426], ignore_index=True)

df['time_us'] = (df['frame'] - 1) * 60
plt.style.use(current_dir.parents[1] / 'default.mplstyle')
diam2color = {40:'#0072b2', 60:'#009e73', 100:'#e69f00', 200:'#cc79a7'}

# first, let's plot the polarization
fig1 = plt.figure(figsize=(4,4), dpi=300)
h = [Size.Fixed(1.0), Size.Fixed(2)]
v = [Size.Fixed(0.7), Size.Fixed(2)]
divider = Divider(fig1, (0, 0, 1, 1), h, v, aspect=False)
axs1 = fig1.add_axes(divider.get_position(),
                     axes_locator=divider.new_locator(nx=1, ny=1))
for name, group in df.groupby('bead_diam_nm'):
    ratio_mean = group.groupby('time_us')['polarization'].mean()
    ratio_std = group.groupby('time_us')['polarization'].std()
    times_us = group.groupby('time_us').groups.keys()
    axs1.plot(times_us, ratio_mean, '.-', markersize=5,
              label=name, color=diam2color[name])
    axs1.fill_between(times_us, ratio_mean - ratio_std,
                      ratio_mean + ratio_std, alpha=0.2,
                      color=diam2color[name])
plt.title('Experiment: mScarlet')
for ax in fig1.axes:
    ax.grid('on', alpha=0.3)
    ax.legend(title='Diameter (nm)', ncol=2, fontsize=8, columnspacing=1.3,
              handlelength=1.3, title_fontsize=8, framealpha=1,
              handletextpad=0.7, borderpad=0.5)
    ax.set_xlabel(r'Pump-Probe Delay Time ($\mu$s)')
    ax.set_ylabel('Polarization')
    ax.set_ylim(-0.05, 0.35)
    ax.set_xlim(-25, 1025)
plt.xticks(range(0, 1250, 250))
plt.savefig('01_mScarlet_polarization.pdf', bbox_inches='tight',
            transparent=True)

# let's also take a look at the anisotropy
fig2 = plt.figure(figsize=(4,4), dpi=300)
h = [Size.Fixed(1.0), Size.Fixed(2)]
v = [Size.Fixed(0.7), Size.Fixed(2)]
divider = Divider(fig2, (0, 0, 1, 1), h, v, aspect=False)
axs2 = fig2.add_axes(divider.get_position(),
                     axes_locator=divider.new_locator(nx=1, ny=1))
for name, group in df.groupby('bead_diam_nm'):
    ratio_mean = group.groupby('time_us')['anisotropy'].mean()
    ratio_std = group.groupby('time_us')['anisotropy'].std()
    times_us = group.groupby('time_us').groups.keys()
    axs2.plot(times_us, ratio_mean, '.-', markersize=5,
              label=name, color=diam2color[name])
    axs2.fill_between(times_us, ratio_mean - ratio_std,
                      ratio_mean + ratio_std, alpha=0.2,
                      color=diam2color[name])
for ax in fig2.axes:
    ax.grid('on', alpha=0.3)
    ax.legend(title='Diameter (nm)', ncol=2, columnspacing=1.3, fontsize=8,
              handlelength=1.3, title_fontsize=8, framealpha=1,
              handletextpad=0.7, borderpad=0.5)
    ax.set_xlabel(r'Pump-Probe Delay Time ($\mu$s)')
    ax.set_ylabel('Anisotropy')
    ax.set_xlim(-25, 1025)
plt.xticks(range(0, 1250, 250))
plt.title('Experiment: mScarlet')
plt.savefig('02_mScarlet_anisotropy.pdf', bbox_inches='tight',
            transparent=True)

plt.show()

