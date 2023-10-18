from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.image as mpimg

# This script generates the main-text figure comparing mScarlet pump-probe
# tumbling measurements on a commercial instrument to a simulation.

# This script assumes that you have already run
# inspect_triplets_delay_series_8 and _16 to extract raw tumbling data.
# It also assumes that you have run the triplet_anisotropy_plotting.py
# preprocessing script on each individual days' worth of data.

# Read in the triplet counts and polarization ratios (already g factor
# corrected) for each day.
cwd = Path(__file__).parents[0]
pdir = cwd.parents[2] / 'tumbling_temp'
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
plt.style.use(cwd.parents[0] / 'default.mplstyle')
diam2color = {40:'#0072b2', 60:'#009e73', 100:'#e69f00', 200:'#cc79a7'}
diff2color = {21746: '#0072b2', 73394: '#009e73', 339789: '#e69f00',
              2718318: '#cc79a7'}

# let's generate the overall figure with both simulation and experiment
fig = plt.figure(figsize=(5.5, 4.5))
grid = gs.GridSpec(2, 2, height_ratios=[1, 1.2])
ax1 = plt.subplot(grid[0, 0]) # bead schematic
ax2 = plt.subplot(grid[0, 1]) # image of instrument
ax3 = plt.subplot(grid[1, 0]) # experimental results
ax4 = plt.subplot(grid[1, 1]) # simulation results
grid.update(left=0.11, right=0.97, bottom=0.1, top=1.0, wspace=0.3, hspace=0.2)
ax1.set_axis_off(); ax2.set_axis_off()
# bring in the images for the first 2 axes
beads = mpimg.imread(cwd/'beads.png')
ax1.imshow(beads, aspect='equal')
sp8 = mpimg.imread(cwd/'Sp8_pictures'/'microscope_image.jpg')
ax1.set_position([0.03, 0.63, 0.45, 0.45*4.5/5.5])
ax2.imshow(sp8, aspect='equal')
# add subpanel labels
ax1.text(-30, -50, 'A', fontsize=12, weight='bold')
ax1.text(850, -50, 'B', fontsize=12, weight='bold')
ax1.text(-30, 490, 'C', fontsize=12, weight='bold')
ax1.text(850, 490, 'D', fontsize=12, weight='bold')
# plot the experimental results on ax3
for name, group in df.groupby('bead_diam_nm'):
    ratio_mean = group.groupby('time_us')['polarization'].mean()
    ratio_std = group.groupby('time_us')['polarization'].std()
    times_us = group.groupby('time_us').groups.keys()
    ax3.plot(times_us, ratio_mean, '.-', markersize=5,
             label=name, color=diam2color[name])
    ax3.fill_between(times_us, ratio_mean - ratio_std,
                     ratio_mean + ratio_std, alpha=0.2,
                     color=diam2color[name])
ax3.set_title('Experiment: mScarlet')
ax3.set_ylim(-0.35, 0.05)
ax3.set_yticks(np.linspace(-0.35, 0.05, 5))
ax3.legend(title='Diameter (nm)', ncol=2, fontsize=8, columnspacing=1.3,
           handlelength=1.3, title_fontsize=8, framealpha=1,
           handletextpad=0.7, borderpad=0.5, loc='upper right')
# import the simulation results and plot on axes 4
cres = pd.read_csv('confocal_pump_probe_crescentFactorsOf5_pump2_probe0p25.csv')
cres['polarization'] = (cres['counts_x']-cres['counts_y'])/(
    cres['counts_x']+cres['counts_y'])
cresS = cres.loc[cres['crescent_saturation'] == 1.25]
for name, group in cresS.groupby('diffusion_time_ns'):
    pol_means = group.groupby('delay_us')['polarization'].mean()
    pol_stds = group.groupby('delay_us')['polarization'].std()
    delays = group.groupby('delay_us').groups.keys()
    ax4.plot(delays, pol_means, '.-', markersize=5, color=diff2color[name])
    ax4.fill_between(delays,
                     pol_means - pol_stds,
                     pol_means + pol_stds,
                     color=diff2color[name], alpha=0.2)
ax4.set_title('Simulation')
ax4.set_ylim(-0.45, 0.35)
# format both experiment and simulation axes
for ax in [ax3, ax4]:
    ax.grid('on', alpha=0.3)
    ax.set_xlabel(r'Pump-Probe Delay ($\mu$s)')
    ax.set_ylabel('Polarization')
    ax.set_xlim(-25, 1025)
    ax.set_xticks(range(0, 1250, 250))
fig.savefig("mScarlet_SP8.png", dpi=300)

plt.show()

