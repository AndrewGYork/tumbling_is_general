from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import Divider, Size

# This script plots how the polarization ratio and anisotropy change for
# the triggered fluorescence from mVenus at increasing trigger delays.
# (In other words, it plots a tumbling curve.)

# This script assumes that you have already run
# inspect_triplets_delay_series_8 and _16 to extract raw tumbling data.
# It also assumes that you have run the triplet_anisotropy_plotting.py
# preprocessing script on each individual days' worth of data.

# Read in the triplet counts and polarization ratios (already g factor
# corrected) for each day.
cwd = Path(__file__).parents[0]
pdir = cwd.parents[3] / 'tumbling_temp'
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
plt.style.use(cwd.parents[1] / 'default.mplstyle')
diam2color = {40:'#0072b2', 60:'#009e73', 100:'#e69f00', 200:'#cc79a7'}

# first, let's plot the polarization ratios
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
plt.title('Experiment: mVenus')
for ax in fig1.axes:
    ax.grid('on')
    ax.legend(title='Diameter (nm)', ncol=2, fontsize=8, columnspacing=1.3,
              handlelength=1.3, title_fontsize=8, framealpha=1,
              handletextpad=0.7, borderpad=0.5)
    ax.set_xlabel(r'Trigger Time ($\mu$s)')
    ax.set_ylabel('Polarization')
    ax.set_ylim(-0.05, 0.35)
    ax.set_xlim(-25, 1025)
plt.xticks(range(0, 1250, 250))
plt.savefig('01_mVenus_polarization.pdf', bbox_inches='tight',
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
    ax.grid('on')
    ax.legend(title='Diameter (nm)', ncol=2, columnspacing=1.3, fontsize=8,
              handlelength=1.3, title_fontsize=8, framealpha=1,
              handletextpad=0.7, borderpad=0.5)
    ax.set_xlabel(r'Trigger Time ($\mu$s)')
    ax.set_ylabel('Anisotropy')
    ax.set_xlim(-25, 1025)
plt.xticks(range(0, 1250, 250))
plt.title('Experiment: mVenus')
plt.savefig('02_mVenus_anisotropy.pdf', bbox_inches='tight',
            transparent=True)

plt.show()

