from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import pandas as pd

# This script visualizes and analyzes the fluorescence anisotropy of the
# prompt ("singlet") signal to evaluate the quality of the sample.

# This script assumes that you have already run all of the preprocessing
# scripts on the raw data (listed below), as it accesses and plots the
# output of those scripts. Please make sure those scripts have been run
# before attempting to plot the results with this script.

# Required pre-processing (all with the raw data in Source Data):
#   inspect_singlet_tcspc.py

# Read in the prompt singlet photons, exported from preprocessing
cwd = Path(__file__).parents[0]
pdir = cwd.parents[4] / 'tumbling_temp'
p0420 = pdir / 'source_data' / '2023-04-20_mScarlet_tumbling.sptw' / 'analysis'
p0424 = pdir / 'source_data' / '2023-04-24_mScarlet_tumbling.sptw' / 'analysis'
p0426 = pdir / 'source_data' / '2023-04-26_mScarlet_tumbling.sptw' / 'analysis'
date_list = [20230420, 20230424, 20230426]
df_list = []
for (p, date) in zip([p0420, p0424, p0426], date_list):
    df = pd.read_csv(p / 'output_singlet_tcspc.csv')
    df['date'] = date
    if 'version' not in df.columns:
        df['version'] = 1
    # These files contain more results than what we want to plot here.
    # Let's isolate the results we want to look at.
    if date == 20230426:
        idx = df.index[(df['bead_diam_nm'] == 200) & (df['version'] < 3)]
        df.drop(idx, inplace=True)
        idx2 = df.index[(df['bead_diam_nm'] == 100) & (df['version'] == 2)]
        df.drop(idx2, inplace=True)
    elif date == 20230420:
        idx = df.index[(df['bead_diam_nm'] == 200) |
                       (df['bead_diam_nm'] == 60)]
        df.drop(idx, inplace=True)
    df_list.append(df)
df = pd.concat(df_list, ignore_index=True)

# Apply each day's g factor to that data
date2g = {20230420: 0.89, 20230424: 0.90, 20230426: 0.90}
for date in date_list:
    g = date2g[date]
    perp = df.loc[df['date'] == date]['detectorA'].values * g
    para = df.loc[df['date'] == date]['detectorB'].values
    with np.errstate(divide='ignore', invalid='ignore'):
        df.loc[df['date']==date, 'pol_ratio'] = para / perp
        df.loc[df['date']==date, 'anisotropy'] = (para - perp) / (para + 2*perp)
        df.loc[df['date']==date, 'polarization'] = (para - perp) / (para + perp)

# Some parameter setup
plt.style.use(cwd.parents[2] / 'default.mplstyle')
nbins = 100 # bins 100-128 are padding
ns_per_bin = 12.5 / nbins
df['time_ns'] = df['time_bin'] * ns_per_bin
diam2color = {40:'#0072b2', 60:'#009e73', 100:'#e69f00', 200:'#cc79a7'}

# plot the individual days' polarization for QC
fig1, axs1 = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8,8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
diam2ax = {40: 0, 60: 1, 100: 2, 200: 3}
beads = df.loc[df['bead_diam_nm'] > 0]
for name, group in beads.groupby(['bead_diam_nm', 'date', 'version']):
    ax = diam2ax[name[0]]
    fig1.axes[ax].plot(group['time_ns'].values,
                       group['polarization'].values,
                       label=str(name[1])+'_'+str(name[2]))
    fig1.axes[ax].set_title(str(name[0]) + 'nm')
for ax in fig1.axes:
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(r'Polarization, ($\parallel - \perp)/(\parallel + \perp$)')
    ax.grid()
    ax.legend()
    ax.set_ylim(0, 0.5)
    ax.set_xlim(0, 12.5)
plt.title('mScarlet')
plt.savefig('01_mScarlet_TR-FA_polarization_by_day.pdf', bbox_inches='tight',
            transparent=True)

# Plot the polarization ratios of the averaged data
beads = df.loc[df['bead_diam_nm'] > 0]
fig2 = plt.figure(figsize=(4,4))
h = [Size.Fixed(1.0), Size.Fixed(2)]
v = [Size.Fixed(0.7), Size.Fixed(2)]
divider = Divider(fig2, (0, 0, 1, 1), h, v, aspect=False)
axs2 = fig2.add_axes(divider.get_position(),
                     axes_locator=divider.new_locator(nx=1, ny=1))
for name, group in beads.groupby('bead_diam_nm'):
    ratio_means = group.groupby('time_ns')['polarization'].mean()
    ratio_stds = group.groupby('time_ns')['polarization'].std()
    delays = group.groupby('time_ns').groups.keys()
    axs2.plot(delays, ratio_means, label=name, color=diam2color[name])
    axs2.fill_between(delays, ratio_means - ratio_stds,
                      ratio_means + ratio_stds, alpha=0.2,
                      color=diam2color[name])
axs2.set_xlim(0.75, 12)
axs2.set_ylim(0, 0.5)
axs2.grid()
axs2.set_ylabel(r'Polarization, ($\parallel$-$\perp$) / ($\parallel$+$\perp$)')
axs2.set_xlabel('Time (ns)')
plt.title('mScarlet')
plt.legend(ncol=2, loc='lower left', title='Diameter (nm)')
plt.savefig('02_mScarlet_TR-FA_averaged_polarization.pdf', bbox_inches='tight',
            transparent=True)

# Plot the anisotropy of the averaged data
fig3 = plt.figure(figsize=(4,4))
h = [Size.Fixed(1.0), Size.Fixed(2)]
v = [Size.Fixed(0.7), Size.Fixed(2)]
divider = Divider(fig3, (0, 0, 1, 1), h, v, aspect=False)
axs3 = fig3.add_axes(divider.get_position(),
                     axes_locator=divider.new_locator(nx=1, ny=1))
for name, group in beads.groupby('bead_diam_nm'):
    ratio_means = group.groupby('time_ns')['anisotropy'].mean()
    ratio_stds = group.groupby('time_ns')['anisotropy'].std()
    delays = group.groupby('time_ns').groups.keys()
    axs3.plot(delays, ratio_means, label=name, color=diam2color[name])
    axs3.fill_between(delays, ratio_means - ratio_stds,
                      ratio_means + ratio_stds, alpha=0.2,
                      color=diam2color[name])
axs3.set_xlim(0.75, 12)
axs3.set_ylim(0, 0.4)
axs3.grid()
axs3.set_ylabel('Anisotropy')
axs3.set_xlabel('Time (ns)')
plt.title('mScarlet')
plt.legend(ncol=2, loc='lower left', title='Diameter (nm)')
plt.savefig('03_mScarlet_TR-FA_averaged_anisotropy.pdf', bbox_inches='tight',
            transparent=True)

plt.show()

