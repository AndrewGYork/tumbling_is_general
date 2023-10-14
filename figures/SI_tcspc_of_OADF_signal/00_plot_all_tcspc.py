from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import Divider, Size
from scipy.optimize import curve_fit

# This script visualizes and analyzes the fluorescence lifetime of the
# prompt ("singlet") signal and the triggered triplet signal to compare
# the two. It uses various conditions, including protein on beads and
# protein in different buffers. It processes data from mScarlet and mVenus.

# This script assumes that you have already run all of the preprocessing
# scripts on the raw data (listed below), as it accesses and plots the
# output of those scripts. Please make sure those scripts have been run
# before attempting to plot the results with this script. It also uses
# pre-processing of the bead data run in subfolders within this main
# folder (combining data across days).

# Required pre-processing (all with the raw data in Source Data):
#   inspect_singlet_tcspc.py
#   inspect_triplet_tcspc_8f.py
#   inspect_triplet_tcspc_16f.py
#   inspect_triplet_tumbling_8f.py
#   inspect_triplet_tumbling_16f.py
#   10_triplet_plotting.py (in the analysis subfolder, calculates background)
#   20_analyze_tcspc_hyd.py

##def calc_mean_bin(decay, axis=0):
##    # decay is an N dimensional array. slowest dimension is TCSPC time by default
##    s = np.ones(len(decay.shape), dtype='uint32')
##    s[axis] = decay.shape[axis]
##    bin_idx = np.arange(decay.shape[axis]).reshape(s)
##    with np.errstate(divide='ignore', invalid='ignore'):
##        result = (decay * bin_idx).sum(axis) / decay.sum(axis)
##    return result
##
##def bins_to_ns(mean_bin_image, zero_bin_idx, ns_per_bin):
##    return (mean_bin_image - zero_bin_idx) * ns_per_bin

# Import all of the HyD data, which is minimally preprocessed
cwd = Path(__file__).parents[0]
sd = cwd.parents[2]/'tumbling_temp'/'source_data'
df0929 = pd.read_csv(sd/'2023-09-29_prompt_vs_triggered'/'analysis'/'summed_singlet_triplet_traces.csv')
df0929['date'] = '20230929'
df0928 = pd.read_csv(sd/'2023-09-28_prompt_vs_triggered'/'analysis'/'summed_singlet_triplet_traces.csv')
df0928['date'] = '20230928'
df0926 = pd.read_csv(sd/'2023-09-26_prompt_vs_triggered_mVenus_PBS'/'analysis'/'summed_singlets_triplets.csv')
df0926['date'] = '20230926'
df0925 = pd.read_csv(sd/'2023-09-25_prompt_vs_triggered_mScarlet_PBS'/'analysis'/'summed_singlets_triplets.csv')
df0925['date'] = '20230925'
df_hyd = pd.concat([df0925, df0926, df0928, df0929], ignore_index=True)

# also read in the 40 nm bead data from SPADs, which is imported
# directly from preprocessing in local subfolders.
S_bead_path = cwd / 'mScarlet_bead_preprocessing' / 'mScarlet_40nmbead_traces.csv'
S_bead_orig = pd.read_csv(S_bead_path)
S_bead_orig['date'] = 'SPAD'
V_bead_path = cwd / 'mVenus_bead_preprocessing' / 'mVenus_40nmbead_traces.csv'
V_bead_orig = pd.read_csv(V_bead_path)
V_bead_orig['date'] = 'SPAD'
df_all = pd.concat([df_hyd, S_bead_orig, V_bead_orig], ignore_index=True)

# Some parameter setup
plt.style.use(cwd.parents[0] / 'default.mplstyle')
nb, bf = (100, 92) # bins 100-128 are padding; the last few bins aren't always high quality
ns_per_bin = 12.5 / nb
time_bin_sequence = range(nb)
time_ns = np.arange(0, 12.5, ns_per_bin)

print('Imported Data for the Following Conditions:')
for n, g in df_all.groupby(['date', 'fluorophore', 'condition']):
    print(n)

# Let's normalize and overlay, keeping the date information where we have duplicates
cond2ax = {'PBS_pH7': 0, 'MES_pH6p3': 1, '40nmbeads': 2, 'Carmody_IB_pH5': 3}
fig1, axs1 = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(10,4))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
for name, group in df_all.groupby(['date', 'fluorophore', 'condition']):
    trip = group.loc[group['intended_signal'] == 'triplet']['counts_triplet_norm']
    sing = group.loc[group['intended_signal'] == 'singlet']['counts_singlet_norm']
    b0_s, b0_t = (np.argmax(sing), np.argmax(trip))    
    shift = b0_t - b0_s
    xs = np.arange((b0_s-5)*ns_per_bin, 11.5, ns_per_bin)
    xt = np.arange((b0_s-5)*ns_per_bin, 11.5 - ns_per_bin*shift, ns_per_bin)
    if name[1] == 'mScarlet':
        a = cond2ax[name[2]]
    elif name[1] == 'mVenus':
        a = cond2ax[name[2]] + 4
    fig1.axes[a].semilogy(xs, sing[(b0_s-5):bf], label='prompt', color='#0066cc')
    fig1.axes[a].semilogy(xt, trip[(b0_t-5):bf], label='triggered', color='#e69900')
    cond_label = {'Carmody_IB_pH5': 'Carmody pH 5', '40nmbeads': '40 nm beads',
                  'MES_pH6p3': 'MES pH 6.3', 'PBS_pH7': 'PBS pH 7'}[name[2]]
    fig1.axes[a].set_title(name[1]+", "+cond_label)
for ax in fig1.axes:
    ax.set_xlabel('Relative Time (ns)')
    ax.set_ylabel('Norm. Counts')
    ax.grid('on', alpha=0.2)
fig1.axes[-1].legend()
fig1.savefig('01_all_overlaid.png', bbox_inches='tight')

# Let's drop the duplicate data. We can't really add it together because
# the offset on the detectors wasn't consistent. Results from different
# days look pretty similar, though.
to_drop = df_hyd.index[(df_hyd['date'] == '20230929') &
                       (df_hyd['condition'] == 'PBS_pH7')]
df_u = df_hyd.drop(to_drop)

# Let's make a figure for the supplementary material, with one replicate
# per condition, all maesured on the HyDs so there's no background
# subtraction to worry about.
cond2ax = {'PBS_pH7': 0, 'MES_pH6p3': 1, '40nmbeads': 2, 'Carmody_IB_pH5': 3}
fig2, axs2 = plt.subplots(2, 4, sharey=True, sharex=True, figsize=(10,4))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
for name, group in df_u.groupby(['date', 'fluorophore', 'condition']):
    trip = group.loc[group['intended_signal'] == 'triplet']['counts_triplet_norm']
    sing = group.loc[group['intended_signal'] == 'singlet']['counts_singlet_norm']
    b0_s, b0_t = (np.argmax(sing), np.argmax(trip))    
    shift = b0_t - b0_s
    xs = np.arange((b0_s-5)*ns_per_bin, 11.5, ns_per_bin)
    xt = np.arange((b0_s-5)*ns_per_bin, 11.5 - ns_per_bin*shift, ns_per_bin)
    if name[1] == 'mScarlet':
        a = cond2ax[name[2]]
    elif name[1] == 'mVenus':
        a = cond2ax[name[2]] + 4
    fig2.axes[a].semilogy(xs, sing[(b0_s-5):bf], label='prompt', color='#0066cc')
    fig2.axes[a].semilogy(xt, trip[(b0_t-5):bf], label='triggered', color='#e69900')
    cond_label = {'Carmody_IB_pH5': 'Carmody pH 5', '40nmbeads': '40 nm beads',
                  'MES_pH6p3': 'MES pH 6.3', 'PBS_pH7': 'PBS pH 7'}[name[2]]
    fig2.axes[a].set_title(name[1]+", "+cond_label)
for ax in fig2.axes:
    ax.set_xlabel('Relative Time (ns)')
    ax.set_ylabel('Norm. Counts')
    ax.grid('on', alpha=0.2)
fig2.axes[-1].legend()
fig2.savefig('02_overlaid_unique.png', bbox_inches='tight', dpi=300)

## Now, let's go by condition and generate a plot with fits for each for
## more detailed analysis
fit_list = []
for name, group in df_u.groupby(['fluorophore', 'condition']):
    fig6, axs6 = plt.subplots(2, 1, figsize=(5,4), height_ratios=[3, 1], sharex=True)
    # Tailfit the lifetime data so that we can be somewhat quantitative
    # about testing the comparison between the triplets and the singlets.
    # Let's get the data to fit pulled together first.
    trip = group.loc[group['intended_signal'] == 'triplet']['counts_triplet']
    sing = group.loc[group['intended_signal'] == 'singlet']['counts_singlet']
    b0_s, b0_t = (np.argmax(sing), np.argmax(trip))       
    o = 2 # offset from the peak
    xs = np.arange((b0_s+o)*ns_per_bin, 11.5, ns_per_bin)
    xt = np.arange((b0_t+o)*ns_per_bin, 11.5, ns_per_bin)
    ys, yt = (sing[(b0_s+o):bf], trip[(b0_t+o):bf])
    # plot the curves so we can overlay the fits on them
    axs6[0].semilogy(np.arange((b0_s-5)*ns_per_bin, 11.5, ns_per_bin),
                     sing[(b0_s-5):bf], label='prompt',
                     color='#b3d9ff')
    axs6[0].semilogy(np.arange((b0_t-5)*ns_per_bin, 11.5, ns_per_bin),
                     trip[(b0_t-5):bf], label='triggered',
                     color='#ffd480')
    print('\nTail Fitting', name)
    if (name[1] == '40nmbeads') or (name[1] == 'Carmody_IB_pH5'):
        # Fit the singlets with a 2 exponential model
        popt, pcov = curve_fit(lambda t,a,b,c,d: a*np.exp(b*t) + c*np.exp(d*t), xs, ys,
                               p0=(200, -0.2, 1000, -0.6))
        a_s = popt[0]; b_s = popt[1]; c_s = popt[2]; d_s = popt[3]
        prompt_resid = ys - (a_s*np.exp(b_s*xs) + c_s*np.exp(d_s*xs))
        perr = np.sqrt(np.diag(pcov))
        fig6.axes[0].semilogy(xs,
                              a_s*np.exp(b_s*xs) + c_s*np.exp(d_s*xs),
                              label='prompt_fit', color='#0066cc')
        t_avg = (a_s*(-1/b_s) + c_s*(-1/d_s)) / (a_s + c_s)
        print('Prompt Weighted Avg Tau (2 exp. fit): %0.4f ns'%(t_avg))
        print('Prompt, 2 exp. components (ns): %0.4f, %0.4f ns'%(-1/b_s,-1/d_s))
        s_string = r'Prompt $\tau$ (weighted avg., 2 exp. fit): %0.1f ns'%(t_avg)
        prompt_fit = pd.DataFrame({'type': 'prompt',
                                   'condition': name[1],
                                   'fluorophore': name[0],
                                   'a1': popt[0], 't1': -1/popt[1],
                                   'a1_std': perr[0], 't1_std': perr[1],
                                   'a2': popt[2], 't2': -1/popt[3],
                                   'a2_std': perr[2], 't2_std': perr[3],
                                   't_avg': t_avg},
                                  index=[0])
    else:
        # Fit the singlets with a 1 exponential model
        popt, pcov = curve_fit(lambda t,a,b: a*np.exp(b*t), xs, ys, p0=(730, -0.2))
        a_s = popt[0]; b_s = popt[1]
        prompt_resid = ys - a_s*np.exp(b_s*xs)
        perr = np.sqrt(np.diag(pcov))
        fig6.axes[0].semilogy(xs, a_s*np.exp(b_s*xs),
                              label='prompt_fit', color='#0066cc')
        print('Prompt Tau (1 exp. fit): %0.4f ns' % (-1/b_s))
        s_string = r'Prompt $\tau$ (1 exp. fit): %0.1f ns' % (-1/b_s)
        prompt_fit = pd.DataFrame({'type': 'prompt',
                                   'condition': name[1],
                                   'fluorophore': name[0],
                                   'a1': popt[0], 't1': -1/popt[1],
                                   'a1_std': perr[0], 't1_std': perr[1]},
                                  index=[0])        
    # Fit the triplet signal to a single exponential. 1 exponential
    # describes the data well; 2 would match the singlet data but would
    # definitely be overfitting here.
    popt, pcov = curve_fit(lambda t,a,b: a*np.exp(b*t), xt, yt, p0=(730, -0.2))
    a_t = popt[0]; b_t = popt[1]
    trig_resid = yt - a_t*np.exp(b_t*xt)
    fig6.axes[0].semilogy(xt,
                          a_t*np.exp(b_t*xt),
                          label='triggered_fit', color='#e69900')
    print('Triggered Tau (1 exp. fit): %0.4f ns' % (-1/b_t))
    t_string = r'Triggered $\tau$ (1 exp. fit): %0.1f ns' % (-1/b_t)
    perr = np.sqrt(np.diag(pcov))
    triggered_fit = pd.DataFrame({'type': 'triggered',
                                  'condition': name[1],
                                  'fluorophore': name[0],                                  
                                  'a1': popt[0], 't1': -1/popt[1],
                                  'a1_std': perr[0], 't1_std': perr[1]},
                                 index=[0])
    fits = pd.concat([prompt_fit, triggered_fit], ignore_index=True)
    fit_list.append(fits)
    # calculate and plot the residuals too
    fig6.axes[1].plot(xs, prompt_resid, label='prompt', color='#0066cc')
    fig6.axes[1].plot(xt, trig_resid, label='triggered', color='#e69900')
    # write the fit results on each plot
    fig6.axes[0].text(0.05, 0.03, t_string, transform=fig6.axes[0].transAxes)
    fig6.axes[0].text(0.05, 0.11, s_string, transform=fig6.axes[0].transAxes)
    for ax in fig6.axes:
        ax.set_xlabel('Time (ns)')
    fig6.axes[0].set_ylabel('Counts')
    fig6.axes[0].legend(loc='upper right')
    fig6.axes[1].set_ylabel('Residuals')
    cond_label = {'Carmody_IB_pH5': 'Carmody pH 5', '40nmbeads': '40 nm beads',
                  'MES_pH6p3': 'MES pH 6.3', 'PBS_pH7': 'PBS pH 7'}[name[1]]
    fig6.axes[0].set_title(name[0]+", "+cond_label)
    plt.savefig('%s_%s_fits.png' % (name),bbox_inches='tight')
fits = pd.concat(fit_list, ignore_index=True)
fits.to_csv('all_fit_parameters.csv')

                                  
plt.show()

