from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import Divider, Size
from scipy.optimize import curve_fit

# This script visualizes and analyzes the fluorescence lifetime of the
# prompt ("singlet") signal and the triggered triplet signal to compare
# the two. It is currently analyzing data from the 40 nm beads labeled
# with mScarlet (all 3 replicates). I selected 40 nm beads for their
# high label density (and therefore high triplet signal). I don't expect
# this result to vary by bead size. (Note that the 'TCSPC' analysis was
# only run on the 40 nm beads' triplets, which is what is selecting the
# 40 nm bead size. There is nothing else about this that is 40
# nm-specific.)

# This script assumes that you have already run all of the preprocessing
# scripts on the raw data (listed below), as it accesses and plots the
# output of those scripts. Please make sure those scripts have been run
# before attempting to plot the results with this script.

# Required pre-processing (all with the raw data in Source Data):
#   inspect_singlet_tcspc.py
#   inspect_triplet_tcspc_8f.py
#   inspect_triplet_tcspc_16f.py
#   inspect_triplet_tumbling_8f.py
#   inspect_triplet_tumbling_16f.py
#   10_triplet_plotting.py (in the analysis subfolder, calculates background)

def calc_mean_bin(decay, axis=0):
    # decay is an N dimensional array. slowest dimension is TCSPC time by default
    s = np.ones(len(decay.shape), dtype='uint32')
    s[axis] = decay.shape[axis]
    bin_idx = np.arange(decay.shape[axis]).reshape(s)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (decay * bin_idx).sum(axis) / decay.sum(axis)
    return result

def bins_to_ns(mean_bin_image, zero_bin_idx, ns_per_bin):
    return (mean_bin_image - zero_bin_idx) * ns_per_bin

# The next ~50 lines of code import preprocessed files
# Read in the triggered signal from 2023-04-20
current_dir = Path(__file__).parents[0]
pdir = current_dir.parents[3] / 'tumbling_temp'
p0420 = pdir / 'source_data' / '2023-04-20_mScarlet_tumbling.sptw' / 'analysis'
tdf0420_1 = pd.read_csv(p0420 / 'output_8f_triplet_tcspc.csv')
tdf0420_2 = pd.read_csv(p0420 / 'output_16f_triplet_tcspc.csv')
tdf0420 = pd.concat([tdf0420_1, tdf0420_2], ignore_index=True)
# Read in the prompt singlet photons too
sdf0420 = pd.read_csv(p0420 / 'output_singlet_tcspc.csv')
sdf0420['date'] = 20230420
drop_me = sdf0420.index[sdf0420['bead_diam_nm'] != 40]
sdf0420.drop(drop_me, inplace=True)
# Also read in the background data
bdf0420 = pd.read_csv(p0420 / '11_2023-04-20_mScarlet_bkgd_traces.csv')
# we are only using the 40 nm beads, which are background groups 7 and 8
bdf0420 = bdf0420.loc[(bdf0420['bkgd_group'] == 7) |
                      (bdf0420['bkgd_group'] == 8)]
drop_me = bdf0420.index[(bdf0420['frame'] < 9) & (bdf0420['bkgd_group'] == 8)]
bdf0420.drop(drop_me, inplace=True)
# Read in the triggered signal from 2023-04-24
p0424 = pdir / 'source_data' / '2023-04-24_mScarlet_tumbling.sptw' / 'analysis'
tdf0424_1 = pd.read_csv(p0424 / 'output_8f_triplet_tcspc.csv')
tdf0424_2 = pd.read_csv(p0424 / 'output_16f_triplet_tcspc.csv')
tdf0424 = pd.concat([tdf0424_1, tdf0424_2], ignore_index=True)
# Read in the prompt singlet photons too
sdf0424 = pd.read_csv(p0424 / 'output_singlet_tcspc.csv')
drop_me = sdf0424.index[sdf0424['bead_diam_nm'] != 40]
sdf0424['date'] = 20230424
sdf0424.drop(drop_me, inplace=True)
# Also read in the background data
bdf0424 = pd.read_csv(p0424 / '11_2023-04-24_mScarlet_bkgd_traces.csv')
# we are only using the 40 nm beads, which are background groups 7 and 8
bdf0424 = bdf0424.loc[(bdf0424['bkgd_group'] == 7) |
                      (bdf0424['bkgd_group'] == 8)]
drop_me = bdf0424.index[(bdf0424['frame'] < 9) & (bdf0424['bkgd_group'] == 8)]
bdf0424.drop(drop_me, inplace=True)
# Read in the triggered signal from 2023-04-26
p0426 = pdir / 'source_data' / '2023-04-26_mScarlet_tumbling.sptw' / 'analysis'
tdf0426_1 = pd.read_csv(p0426 / 'output_8f_triplet_tcspc.csv')
tdf0426_2 = pd.read_csv(p0426 / 'output_16f_triplet_tcspc.csv')
tdf0426 = pd.concat([tdf0426_1, tdf0426_2], ignore_index=True)
# Read in the prompt singlet photons too
sdf0426 = pd.read_csv(p0426 / 'output_singlet_tcspc.csv')
drop_me = sdf0426.index[sdf0426['bead_diam_nm'] != 40]
sdf0426['date'] = 20230426
sdf0426.drop(drop_me, inplace=True)
# Also read in the background data
bdf0426 = pd.read_csv(p0426 / '11_2023-04-26_mScarlet_bkgd_traces.csv')
# we are only using the 40 nm beads, which are background groups 5 and 6
bdf0426 = bdf0426.loc[(bdf0426['bkgd_group'] == 5) |
                      (bdf0426['bkgd_group'] == 6)]
drop_me = bdf0426.index[(bdf0426['frame'] < 9) & (bdf0426['bkgd_group'] == 6)]
bdf0426.drop(drop_me, inplace=True)

# Some parameter setup
plt.style.use(current_dir.parents[1] / 'default.mplstyle')
nb = 100 # bins 100-128 are padding
ns_per_bin = 12.5 / nb
time_bin_sequence = range(nb)
time_ns = np.arange(0, 12.5, ns_per_bin)

# The triplet data still need some more processing; we need to extract
# signal only from the frames when we actually triggered triplets.
tcspc_list = []
mean_tau_list = []
for signal, bkgd, date in zip([tdf0420, tdf0424, tdf0426],
                              [bdf0420, bdf0424, bdf0426],
                              [20230420, 20230424, 20230426]):
    print('\n\nProcessing dataset from', date, end='')
    # Get the overall peak frame so that the mean bin calculation
    # is more accurate.
    summed_signal = np.zeros(128)
    for name, group in signal.groupby('trigger_frame'):
        summed_signal += group['detectorA'+str(name-1)].values
        summed_signal += group['detectorB'+str(name-1)].values
    b0 = np.argmax(np.diff(summed_signal))
##    # Some optional code for inspecting the zero bin calling
##    plt.figure()
##    plt.plot(summed_signal)
##    plt.axvline(b0, color='#808080', label='0 bin')
##    plt.title(str(date) + ' Summed Triggered Signal')
##    plt.xlabel('Counts'); plt.ylabel('Time Bin')

    fig1, axs1 = plt.subplots(2, 2, figsize=(8,8), sharex=True, sharey=True)
    A = np.zeros(nb); B = np.zeros(nb)
    for name, group in signal.groupby('trigger_frame'):
        print('.', end='')
        assert len(group) == 128 # only 1 recording
        a = 0 if name < 9 else 2 # axis
        # read in & remove the dark count background per time pixel
        total_bkgd = bkgd.loc[bkgd['frame'] == name]
        assert len(total_bkgd) == 1 # we successfully identified 1 value
        A_bkgd_per_bin = total_bkgd['bkgd_A'].values / nb
        B_bkgd_per_bin = total_bkgd['bkgd_B'].values / nb
        A += group['detectorA'+str(name-1)].values[:nb] - A_bkgd_per_bin
        B += group['detectorB'+str(name-1)].values[:nb] - B_bkgd_per_bin
        fig1.axes[a].semilogy(group['time_bin'][:nb]*ns_per_bin,
                              group['detectorA'+str(name-1)][:nb],
                              label='A_'+str(name))
        fig1.axes[a+1].semilogy(group['time_bin'][:nb]*ns_per_bin,
                                group['detectorB'+str(name-1)][:nb],
                                label='B_'+str(name))
        # let's calculate the mean arrival time of this curve
        f = str(name-1)
        o = A_bkgd_per_bin + B_bkgd_per_bin
        total = group['detectorA'+f].values + group['detectorB'+f].values - o
        mean_bin = calc_mean_bin(total)
        mean_arrival = bins_to_ns(mean_bin, b0, ns_per_bin)
        df0 = pd.DataFrame({'frame': name, 'mean_arrival': mean_arrival,
                            'date': date}, index=[0])
        mean_tau_list.append(df0)
    # save the summed triplets for later analysis
    df1 = pd.DataFrame({'time_bin': range(nb), 'A': A, 'B': B, 'date': date,
                        'time_ns': time_ns})
    tcspc_list.append(df1)
    # figure formatting and saving
    fig1.axes[0].set_title(r'Detector A, $\parallel$ to 775')
    fig1.axes[1].set_title(r'Detector B, $\perp$ to 775')
    for ax in fig1.axes:
        ax.legend()
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Counts')
    plt.suptitle('%d Triggered TCSPC, 40 nm, mScarlet' % date)
    plt.savefig('01_triplets_tcspc_decays_%d.pdf' % date,
                transparent=True,
                bbox_inches='tight')

# concatenate into two dataframes
df = pd.concat(tcspc_list, ignore_index=True)
df.to_csv('02_summed_triplets_40nm.csv')
df_mean_tau = pd.concat(mean_tau_list, ignore_index=True)
df_mean_tau['delay_us'] = (df_mean_tau['frame']-1)* 60
df_mean_tau.to_csv('03_mean_arrival_by_frame_40nm.csv')

# Let's check for any frame dependence in the mean arrival time of the
# photons. We don't expect any, but if we see some it's an indication of
# either (1) more complicated photophysics or (2) instrumentation issues
fig2 = plt.figure()
axs2 = plt.gca()
for name, group in df_mean_tau.groupby('date'):
    fig2.axes[0].plot(group['delay_us'], group['mean_arrival'], '.-', label=name)
for ax in fig2.axes:
    ax.grid()
    ax.legend()
    ax.set_xlabel(r'Delay ($\mu$s)')
    ax.set_ylabel('Mean Arrival Time (ns)')
plt.savefig('04_mean_arrival_vs_trigger_frame.pdf', transparent=True,
            bbox_inches='tight')

# Let's plot all of the triggered triplet signals (raw traces) from the
# two days, both normalized and not, to visually compare them
fig3, axs3 = plt.subplots(1, 2, sharex=True)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
for name, group in df.groupby('date'):
    fig3.axes[0].plot(group['time_ns'], group['A'], label='A_'+str(name))
    fig3.axes[0].plot(group['time_ns'], group['B'], label='B_'+str(name))
    # normalize the data
    a_max = group['A'].max()
    b_max = group['B'].max()
    fig3.axes[1].semilogy(group['time_ns'], group['A']/a_max,
                          label='A_'+str(name))
    fig3.axes[1].semilogy(group['time_ns'], group['B']/b_max,
                          label='B_'+str(name))
for ax in fig3.axes:
    ax.grid()
    ax.legend()
    ax.set_xlabel('Time (ns)')
fig3.axes[0].set_ylabel('Counts')
fig3.axes[1].set_ylabel('Norm. Counts')
plt.suptitle('Triggered Triplets TCSPC, 40 nm, mScarlet')
plt.savefig('05_triplets_tcspc_decays_by_date.pdf', transparent=True,
            bbox_inches='tight')

# Since the decays from the two days look very similar, we can add them
# and proceed with the photon counts combined.
A_tot = df.groupby('time_ns')['A'].sum().values
B_tot = df.groupby('time_ns')['B'].sum().values
trip_tot = A_tot + B_tot

# Similarly, let's examine the signals from both days' singlets and see
# how they compare
sdf = pd.concat([sdf0420, sdf0424, sdf0426], ignore_index=True)
sdf['time_ns'] = sdf['time_bin'] * ns_per_bin
fig4, axs4 = plt.subplots(1, 2, sharex=True)
plt.subplots_adjust(hspace=0.3, wspace=0.3)
for name, group in sdf.groupby('date'):
    fig4.axes[0].plot(group['time_ns'][:nb],
                      group['detectorA'][:nb], label='A_'+str(name))
    fig4.axes[0].plot(group['time_ns'][:nb],
                      group['detectorB'][:nb], label='B_'+str(name))
    # normalize the data
    a_max = group['detectorA'].max()
    b_max = group['detectorB'].max()
    fig4.axes[1].semilogy(group['time_ns'], group['detectorA']/a_max,
                          label='A_'+str(name))
    fig4.axes[1].semilogy(group['time_ns'], group['detectorB']/b_max,
                          label='B_'+str(name))
for ax in fig4.axes:
    ax.grid()
    ax.legend()
    ax.set_xlabel('Time (ns)')
    ax.set_xlim(0, 12.5)
fig4.axes[0].set_ylabel('Counts')
fig4.axes[1].set_ylabel('Norm. Counts')
plt.suptitle('Singlets TCSPC, 40 nm, mScarlet')
plt.savefig('06_singlets_tcspc_decays_by_date.pdf', transparent=True,
            bbox_inches='tight')

# Since the decays from the two days look very similar, we can add them
# and proceed with the photon counts combined.
As_tot = sdf.groupby('time_ns')['detectorA'].sum().values[:nb]
Bs_tot = sdf.groupby('time_ns')['detectorB'].sum().values[:nb]
sing_tot = As_tot + Bs_tot

# let's save out the combined signals for other analyses
singlets = pd.DataFrame({'time_ns': time_ns[:nb],
                         'counts_singlet': sing_tot,
                         'counts_singlet_norm': sing_tot / np.max(sing_tot),
                         'fluorophore': 'mScarlet',
                         'condition': '40nmbeads',
                         'intended_signal': 'singlet'})
triplets = pd.DataFrame({'time_ns': time_ns[:nb],
                         'counts_triplet': trip_tot,
                         'counts_triplet_norm': trip_tot / np.max(trip_tot),
                         'fluorophore': 'mScarlet',
                         'condition': '40nmbeads',
                         'intended_signal': 'triplet'})
to_save = pd.concat([singlets, triplets], ignore_index=True)
to_save.to_csv('mScarlet_40nmbead_traces.csv')

# Let's normalize and overlay the signals from the prompt and the
# triggered emission so we can visually compare them. First, we'll
# determine the mean arrival time.
# NOTE THAT THE MEAN ARRIVAL TIME WILL BE ARTIFICIALLY SKEWED TOWARDS
# SHORTER LIFETIME, ESPECIALLY FOR THE TRIPLETS, SINCE WE ARE
# IGNORING/CLIPPING/COUNTING AT THE BEGINNING ANY PHOTONS ARRIVING MORE
# THAN ~9 NS AFTER INITIAL PULSE.
mean_bin_s = calc_mean_bin(sing_tot)
b0_s = np.argmax(sing_tot)
mean_arrival_s = bins_to_ns(mean_bin_s, b0_s, ns_per_bin)
print('\n\nMean Arrival Times')
print('Caveat: Truncation - we are not seeing a full decay, esp. for triplets')
print('Prompt Mean Arrival (ns):', round(mean_arrival_s, 3))
mean_bin_t = calc_mean_bin(trip_tot)
b0_t = np.argmax(trip_tot)
mean_arrival_t = bins_to_ns(mean_bin_t, b0_t, ns_per_bin)
print('Triggered Mean Arrival (ns):', round(mean_arrival_t, 3))
# Now, we'll move on to plotting...
trip_tot_norm = trip_tot / np.max(trip_tot)
sing_tot_norm = sing_tot / np.max(sing_tot)
fig5 = plt.figure(figsize=(4,4), dpi=300)
h = [Size.Fixed(1.0), Size.Fixed(2)]
v = [Size.Fixed(0.7), Size.Fixed(2)]
divider = Divider(fig5, (0, 0, 1, 1), h, v, aspect=False)
axs5 = fig5.add_axes(divider.get_position(),
                     axes_locator=divider.new_locator(nx=1, ny=1))
bf=92
shift = b0_t - b0_s
xs = np.arange((b0_s-5)*ns_per_bin, 11.5, ns_per_bin)
xt = np.arange((b0_s-5)*ns_per_bin, 11.5 - ns_per_bin*shift, ns_per_bin)
plt.semilogy(xs, sing_tot_norm[(b0_s-5):bf], label='prompt', color='#0066cc')
plt.semilogy(xt, trip_tot_norm[(b0_t-5):bf], label='triggered', color='#e69900')
for ax in fig5.axes:
    ax.legend()
    ax.set_xlabel('Relative Time (ns)')
    ax.set_ylabel('Norm. Counts')
plt.title('mScarlet, 40 nm beads')
plt.savefig('mScarlet_40nmbeads_overlaid.png', bbox_inches='tight')

# Let's do a little bit of curve fitting to get at the differences
# between these signals a bit more precisely.
fig6, axs6 = plt.subplots(2, 1, figsize=(5,4), height_ratios=[3, 1],
                          sharex=True)
# Tailfit the lifetime data so that we can be somewhat quantitative
# about testing the comparison between the triplets and the singlets.
# Let's get the data to fit pulled together first.
bf = 92 # clip off the last ns of the trace
o = 2 # offset from the peak
xs = np.arange((b0_s+o)*ns_per_bin, 11.5, ns_per_bin)
ys = sing_tot[(b0_s+o):bf]
xt = np.arange((b0_t+o)*ns_per_bin, 11.5, ns_per_bin)
yt = trip_tot[(b0_t+o):bf]
axs6[0].semilogy(np.arange((b0_s-5)*ns_per_bin, 11.5, ns_per_bin),
                 sing_tot[(b0_s-5):bf], label='prompt',
                 color='#b3d9ff')
axs6[0].semilogy(np.arange((b0_t-5)*ns_per_bin, 11.5, ns_per_bin),
                 trip_tot[(b0_t-5):bf], label='triggered',
                 color='#ffd480')
# Fit the singlets with a 2 exponential model (even with this, there's
# still some structure in the residuals...)
popt, pcov = curve_fit(lambda t,a,b,c,d: a*np.exp(b*t) + c*np.exp(d*t), xs, ys,
                       p0=(200, -0.2, 1000, -0.6))
a_s = popt[0]; b_s = popt[1]; c_s = popt[2]; d_s = popt[3]
perr = np.sqrt(np.diag(pcov))
fig6.axes[0].semilogy(xs,
                      a_s*np.exp(b_s*xs) + c_s*np.exp(d_s*xs),
                      label='prompt_fit', color='#0066cc')
print('\nTail Fitting Results')
print('Prompt Weighted Avg Tau (2 exp. fit):',
      round((a_s*(-1/b_s) + c_s*(-1/d_s)) / (a_s + c_s), 4))
print('Prompt, 2 exp. components (ns):', round(-1/b_s, 4), round(-1/d_s, 4))
prompt_fit = pd.DataFrame({'type': 'prompt',
                           'a1': popt[0], 't1': -1/popt[1],
                           'a1_std': perr[0], 't1_std': perr[1],
                           'a2': popt[2], 't2': -1/popt[3],
                           'a2_std': perr[2], 't2_std': perr[3],
                           't_avg': (a_s*(-1/b_s) + c_s*(-1/d_s)) / (a_s + c_s)},
                          index=[0])
# Fit the triplet signal to a single exponential. 1 exponential
# describes the data well; 2 would match the singlet data but would
# definitely be overfitting here.
popt, pcov = curve_fit(lambda t,a,b: a*np.exp(b*t), xt, yt,
                       p0=(730, -0.2))
a_t = popt[0]; b_t = popt[1]
fig6.axes[0].semilogy(xt,
                      a_t*np.exp(b_t*xt),
                      label='triggered_fit', color='#e69900')
print('Triggered Tau (1 exp fit):', round(-1/b_t, 4))
perr = np.sqrt(np.diag(pcov))
triggered_fit = pd.DataFrame({'type': 'triggered',
                              'a1': popt[0],
                              't1': -1/popt[1],
                              'a1_std': perr[0],
                              't1_std': perr[1]}, index=[0])
fits = pd.concat([prompt_fit, triggered_fit], ignore_index=True)
fits.to_csv('09_fit_parameters.csv')
# calculate and plot the residuals too
prompt_resid = ys - (a_s*np.exp(b_s*xs) + c_s*np.exp(d_s*xs))
trig_resid = yt - a_t*np.exp(b_t*xt)
fig6.axes[1].plot(xs, prompt_resid, label='prompt', color='#0066cc')
fig6.axes[1].plot(xt, trig_resid, label='triggered', color='#e69900')
for ax in fig6.axes:
    ax.set_xlabel('Time (ns)')
fig6.axes[0].set_ylabel('Counts')
fig6.axes[0].legend()
fig6.axes[1].set_ylabel('Residuals')
fig6.axes[0].set_title('mScarlet, 40 nm beads')
plt.savefig('mScarlet_40nmbeads_fits.png', bbox_inches='tight')


plt.show()

