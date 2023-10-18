import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size

# This script plots the results of the 01_simulate_SP8.py script. It
# should be run after that simulation has been completed.

# Some parameter setup
diff2diam = {21746: 40, 73394: 60, 339789: 100, 2718318: 200}
diff2color = {21746: '#0072b2', 73394: '#009e73', 339789: '#e69f00',
              2718318: '#cc79a7'}
cwd = Path.cwd()
plt.style.use(cwd.parents[0] / 'default.mplstyle')
# Read the outputs from the simulation script. I ran it multiple times
# to get higher numbers of replicates, so we are combining these
# results. This is equivalent to running the script once with nrep=12.
cap = pd.read_csv('sp8_simulation_pump2_probe0p25.csv')
cres = pd.read_csv('confocal_pump_probe_crescentFactorsOf5_pump2_probe0p25.csv')
cap['polarization'] = (cap['counts_x']-cap['counts_y'])/(cap['counts_y']+cap['counts_x'])
cres['polarization'] = (cres['counts_x']-cres['counts_y'])/(cres['counts_y']+cres['counts_x'])

# EFFECT OF CRESCENT SELECTION ON THE DYNAMIC RANGE
# We only ran 6 replicates at each crescent condition, so let's just
# combine in with the first 6 replicates of the cap selection data for
# consistency.
cap['crescent_saturation'] = 0
cap1 = cap.loc[cap['replicate'] < 6]
combo = pd.concat([cap1, cres], ignore_index=True)
diff2ax = {21746: 0, 73394: 1, 339789: 2, 2718318: 3}
sat2color = {0: 0, 0.01: 0.2, 0.05: 0.4, 0.25: 0.6, 1.25: 0.8, 6.25: 0.99}
fig2, axs2 = plt.subplots(2, 2, figsize=(6, 5), dpi=300)
plt.subplots_adjust(hspace=0.45, wspace=0.3)
for name, group in combo.groupby(['diffusion_time_ns', 'crescent_saturation']):
    a = diff2ax[name[0]]
    color = plt.cm.Wistia(sat2color[name[1]])
    pol_means = group.groupby('delay_us')['polarization'].mean()
    pol_stds = group.groupby('delay_us')['polarization'].std()
    delays = group.groupby('delay_us').groups.keys()
    fig2.axes[a].plot(delays, pol_means, '.-', markersize=5, label=name[1],
                      color=color)
    fig2.axes[a].fill_between(delays, pol_means - pol_stds,
                              pol_means + pol_stds, alpha=0.2,
                              color=color)
    fig2.axes[a].set_title(str(diff2diam[name[0]]) + ' nm')
fig2.axes[0].legend(title='Crescent Pulse Intensity', ncol=2, fontsize=8,
                    columnspacing=1.3, handlelength=1.3, title_fontsize=8,
                    framealpha=1, handletextpad=0.7, borderpad=0.5)
for ax in fig2.axes:
    ax.grid('on', alpha=0.3)
    ax.set_xlabel(r'Pump-Probe Delay Time ($\mu$s)')
    ax.set_ylabel('Polarization')
    ax.set_xlim(-25, 1025)
    ax.set_ylim(-0.5, 0.5)
plt.xticks(range(0, 1250, 250))
plt.savefig('simulation_varyingCrescent_pump2_probe0p25.pdf', bbox_inches='tight',
            transparent=True)

# EFFECT OF CRESCENT SELECTION ON TOTAL COUNTS
# Let's also plot the total counts relative to the starting counts so
# that we can attempt to relate to the experimental saturation (caveat
# here is of course that the dipole may rotate in expt but not here)
fig3 = plt.figure(figsize=(4,4))
h = [Size.Fixed(1.0), Size.Fixed(1.7)]
v = [Size.Fixed(0.7), Size.Fixed(1.7)]
divider = Divider(fig3, (0, 0, 1, 1), h, v, aspect=False)
axs3 = fig3.add_axes(divider.get_position(),
                     axes_locator=divider.new_locator(nx=1, ny=1))
combo['total_cts'] = combo['counts_x'] + combo['counts_y']
for name, group in combo.groupby('diffusion_time_ns'):
    color = diff2color[name]
    d360 = group.loc[group['delay_us']==360]
    max_cts = d360.loc[d360['crescent_saturation'] == 0]['total_cts'].sum()
    axs3.plot(d360.crescent_saturation.unique(),
              d360.groupby('crescent_saturation')['total_cts'].sum()/max_cts,
              '.-', label=diff2diam[name],
              color=color, markersize=3)
axs3.legend(title='Diameter (nm)')
axs3.grid('on')
axs3.set_ylim(0.1, 1.05)
axs3.set_xlabel('Crescent Saturation Per Pulse')
axs3.set_ylabel(r'Norm. Counts at Delay 360 $\mu$s')
axs3.set_title('Simulation: Crescent')
plt.savefig('signal_loss_from_crescent.pdf', bbox_inches='tight',
            transparent=True)

plt.show()
