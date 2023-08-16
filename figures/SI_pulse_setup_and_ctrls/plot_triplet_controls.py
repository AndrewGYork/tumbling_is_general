from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size

## This script plots the raw signal for sample single trial recordings
## of the tumbling settings with the resonant galvo. The objectives are to
## (1) give a more concrete example of how the data were collected and
## (2) include pump only and probe only controls to illustrate that the
## triplet triggering is behaving as expected.

# Read in the relevant source file
cwd = Path(__file__).parents[0]
source = cwd.parents[2]/'tumbling_temp'/'source_data'/'2023-05-08_controls.sptw'
df = pd.read_csv(source / 'analysis' / 'output_8f_triplet_tumbling.csv')

# Do some formatting and setup
df['time_us'] = (df['frame'] - 1) * 60
plt.style.use(cwd.parents[0] / 'default.mplstyle')

# Make a plot of the mScarlet traces
mScarlet = df.loc[df['fluorophore'] == 'mScarlet']
fig1 = plt.figure(figsize=(4,4))
h = [Size.Fixed(1.0), Size.Fixed(2)]
v = [Size.Fixed(0.7), Size.Fixed(2)]
divider = Divider(fig1, (0, 0, 1, 1), h, v, aspect=False)
axs1 = fig1.add_axes(divider.get_position(),
                     axes_locator=divider.new_locator(nx=1, ny=1))
powers2label = {(30, 2): 'Pump+Probe', (0, 2): 'Probe Only', (30, 0): 'Pump Only'}
for name, group in mScarlet.groupby(['WLL_power', 'trigger_power']):
    axs1.plot(group['time_us'][1:],
              group['detectorA'][1:] + group['detectorB'][1:], '.-',
              markersize=3, label=powers2label[name], linewidth=0.75)
axs1.grid(alpha=0.3)
l = axs1.legend(loc='upper right', bbox_to_anchor=(1.15,1))
l.get_frame().set_linewidth(0.75)
axs1.set_ylabel(r'Total Counts ($\parallel+\perp$)')
axs1.set_xlabel(r'Delay Time ($\mu$s)')
axs1.spines[:].set_linewidth(0.75)
axs1.tick_params(width=0.75)
plt.title(r'mScarlet, Probe Delay 240 $\mu$s', fontsize=8)
plt.savefig('01_mScarlet_probeDelay240us.pdf', transparent=True,
            bbox_inches='tight')

# Make a plot of the mVenus traces
mVenus = df.loc[df['fluorophore'] == 'mVenus']
fig2 = plt.figure(figsize=(4,4))
h = [Size.Fixed(1.0), Size.Fixed(2)]
v = [Size.Fixed(0.7), Size.Fixed(2)]
divider = Divider(fig2, (0, 0, 1, 1), h, v, aspect=False)
axs2 = fig2.add_axes(divider.get_position(),
                     axes_locator=divider.new_locator(nx=1, ny=1))
powers2label = {(60, 2): 'Pump+Probe', (0, 2): 'Probe Only', (60, 0): 'Pump Only'}
for name, group in mVenus.groupby(['WLL_power', 'trigger_power']):
    axs2.plot(group['time_us'][1:],
              group['detectorA'][1:] + group['detectorB'][1:], '.-',
              markersize=3, label=powers2label[name], linewidth=0.75)
axs2.grid(alpha=0.3)
l = axs2.legend(loc='upper right', bbox_to_anchor=(1.15,1))
l.get_frame().set_linewidth(0.75)
axs2.set_ylabel(r'Total Counts ($\parallel+\perp$)')
axs2.set_xlabel(r'Delay Time ($\mu$s)')
axs2.spines[:].set_linewidth(0.75)
axs2.tick_params(width=0.75)
plt.title(r'mVenus, Probe Delay 240 $\mu$s', fontsize=8)
plt.savefig('02_mVenus_probeDelay240us.pdf', transparent=True,
            bbox_inches='tight')

plt.show()
