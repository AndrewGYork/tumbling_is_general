from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import Divider, Size

# This script calculates a normalized decay in triggered counts for
# mVenus at increasing probe delay times. We use this as a proxy for
# determining the lifetime of the (putative) triplet state.

# This script assumes that you have already run
# inspect_triplets_delay_series_8 and _16 to extract raw tumbling data.
# It also assumes that you have run the triplet_anisotropy_plotting.py
# preprocessing script on each individual days' worth of data.

# Read in the triplet counts and polarization ratios (already g factor
# corrected) for each day.
current_dir = Path(__file__).parents[0]
pdir = current_dir.parents[3] / 'tumbling_temp'
p0406 = pdir / 'source_data' / '2023-04-06_Venus_tumbling.sptw' / 'analysis'
df0406 = pd.read_csv(p0406 / '15_2023-04-06_CML_mVenus_triggered_triplets.csv')
df0406['date'] = 20230406
p0410 = pdir / 'source_data' / '2023-04-10_Venus_tumbling.sptw' / 'analysis'
df0410 = pd.read_csv(p0410 / '15_2023-04-10_CML_mVenus_triggered_triplets.csv')
df0410['date'] = 20230410
p0411 = pdir / 'source_data' / '2023-04-11_Venus_tumbling.sptw' / 'analysis'
df0411 = pd.read_csv(p0411 / '15_2023-04-11_CML_mVenus_triggered_triplets.csv')
df0411['date'] = 20230411
p0413 = pdir / 'source_data' / '2023-04-13_Venus_tumbling.sptw' / 'analysis'
df0413 = pd.read_csv(p0413 / '15_2023-04-13_CML_mVenus_triggered_triplets.csv')
df0413['date'] = 20230413
# 04-13 also included some data playing with crescent selection strength; drop that
cres = df0413.index[df0413['crescent_power'] != 0.5]
df0413.drop(cres, inplace=True)
df = pd.concat([df0406, df0410, df0411, df0413], ignore_index=True)

df['time_us'] = (df['frame'] - 1) * 60
plt.style.use(current_dir.parents[1] / 'default.mplstyle')
diam2color = {40:'#0072b2', 60:'#009e73', 100:'#e69f00', 200:'#cc79a7'}

# Triplet lifetime by bead size
# Calculate the normalized counts for each bead size & day
for size in [40, 60, 100, 200]:
    for date in [20230406, 20230410, 20230411]:
        sample = df.loc[(df['bead_diam_nm'] == size) & (df['date'] == date)]
        cts = sample['A_bkgd_sub'].values + sample['B_bkgd_sub'].values
        max_cts = np.max(cts)
        df.loc[(df['bead_diam_nm'] == size) & (df['date'] == date),
               'norm_cts'] = cts / max_cts
# Now we can plot these normalized counts
fig1 = plt.figure(figsize=(4,4), dpi=300)
h = [Size.Fixed(1.0), Size.Fixed(2)]
v = [Size.Fixed(0.7), Size.Fixed(2)]
divider = Divider(fig1, (0, 0, 1, 1), h, v, aspect=False)
axs1 = fig1.add_axes(divider.get_position(),
                     axes_locator=divider.new_locator(nx=1, ny=1))
for name, group in df.groupby('bead_diam_nm'):
    cts_mean = group.groupby('time_us')['norm_cts'].mean()
    cts_std = group.groupby('time_us')['norm_cts'].std()
    times_us = group.groupby('time_us').groups.keys()
    axs1.plot(times_us, cts_mean, '.-', markersize=5,
              label=name, color=diam2color[name])
    axs1.fill_between(times_us, cts_mean - cts_std,
                      cts_mean + cts_std, alpha=0.2,
                      color=diam2color[name])
for ax in fig1.axes:
    ax.grid('on')
    ax.legend(title='Diameter (nm)', ncol=2, columnspacing=1.3, fontsize=8,
              handlelength=1.3, title_fontsize=8, framealpha=1,
              handletextpad=0.7, borderpad=0.5, loc='lower left')
    ax.set_xlabel(r'Pump-Probe Delay Time ($\mu$s)')
    ax.set_ylabel(r'Normalized $\parallel + \perp$ Counts')
    ax.set_ylim(0.1, 1.1)
    ax.set_xlim(-25, 1025)
plt.xticks(range(0, 1250, 250))
plt.title('mVenus')
plt.savefig('01_mVenus_normalized_counts.pdf', bbox_inches='tight',
            transparent=True)

plt.show()

