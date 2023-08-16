import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
from pathlib import Path
import pandas as pd
from fluorophore_rotational_diffusion import Fluorophores, FluorophoreStateInfo
from numpy.random import uniform
from math import floor

# This script simulates the use of polarized photoswitching light to
# etch out an oriented subset of molecules without using any 'fancy'
# photophysics. We are also simulating very minimal hardware here
# (continuous illumination without time-resolved detection).

# Even without time resolution on the detection side, we can measure a
# tumbling spectrum by varying the illumination saturation, which
# changes how quickly the population is bleached relative to how quickly
# it rotates.

# Here, we simulate rotational correlation times that roughly match an
# antibody bound to various sizes of protein complex, imagining a use
# case where a labelled antibody is mixed with a clinical sample &
# protein complex sizes are determined.

def get_xy_emission_counts(population, initial_state, final_state,
                           start_time_ns=0):
    x, y, z, t = population.get_xyzt_at_transitions(initial_state, final_state)
    # to only look at counts from the most recent event, set a start time > 0
    idx = t > start_time_ns
    x_t = x[idx]; y_t = y[idx]
    p_x, p_y = x_t**2, y_t**2 # Probabilities of landing in channel x or y
    r = uniform(0, 1, size=len(x_t))
    in_channel_x = (r < p_x)
    in_channel_y = (p_x <= r) & (r < p_x + p_y)
    return sum(in_channel_x), sum(in_channel_y)

# Some path setup
cwd = Path(__file__).parents[0]
out_dir = cwd.parents[3] / 'tumbling_temp' / 'photoswitching' / 'dots_on_a_sphere'
out_dir.mkdir(exist_ok=True, parents=True)

# Conditions we will test
diffusion_times = [750, 1200, 3e3, 6e3] # antibody + (monomer, small-med-large agg.)
saturations = [0.01, 0.05, 0.25, 1.25, 6.25]
n_fluorophores = [1e2]
nreps = 6

# Let's begin the simulations
print("Simulation: Photoswitching, Off-Switching Saturation")
results_list = []; cdf_list = []
for dtime in diffusion_times:
    for sat in saturations:
        for n_fluor in n_fluorophores:
            for rep in range(nreps):
                print('\n\nDiff. time %0.1e, saturation %0.3f' % (dtime, sat))
                print('{:0.0f} Fluorophores, Replicate {:d}'.format(n_fluor,rep))
                out_name = "photoswitching_dt%0.1e_sat%0.3f_%d"%(dtime,sat,n_fluor)
                out_name = out_name.replace('.', 'p')
                # make an animation frame and track CDF if relevant
                if sat == 1.25 and dtime in [750, 6e3] and rep==0:
                    animating = True 
                    x_cdf = []; y_cdf = []; time_list = []
                    x_cts = 0; y_cts = 0
                    # set up the figure we will put points on
                    # outside the time_evolve loop so points accumulate
                    fig = plt.figure(figsize=(6, 6))
                    ax = plt.subplot(1, 1, 1, projection='3d')
                    ax.computed_zorder=False
                    ax.view_init(10, 20, roll=None, vertical_axis="y")
                    ax.set_xlim(-1.05, 1.05)
                    ax.set_ylim(-1.05, 1.05)
                    ax.set_zlim(-1.05, 1.05)
                    ax.invert_xaxis(); ax.invert_yaxis()
                    ax.set_box_aspect((1, 1, 1))
                    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                        axis.set_ticklabels([])
                        axis._axinfo['juggled'] = (1, 1, 1)
                        axis._axinfo['tick']['inward_factor'] = 0
                        axis._axinfo['tick']['outward_factor'] = 0
                        axis._axinfo['grid']['color'] = '#dcdcdc'
                        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))                                    
                else:
                    animating = False
                # now let's start simulating
                state_info = FluorophoreStateInfo()
                state_info.add('inactive_ground') # inactive state, index 0
                state_info.add('inactive_excited',
                               final_states='active_ground',
                               lifetime=1000)
                state_info.add('active_ground')
                state_info.add('active_excited', lifetime=1,
                               final_states=['active_ground', 'inactive_ground'],
                               probabilities=[0.98, 0.02])
                n, _ = state_info.n_and_lifetime('inactive_ground')
                assert n[0] == 0 # check that inactive ground is index 0
                a = Fluorophores(n_fluor,
                                 diffusion_time=dtime,
                                 state_info=state_info,
                                 initial_state=0) # assume we inactivated first
                print('Recording', end='')
                # Perpendicular activation and excitation/off-switching
                for i in range(10000):
                    if i % 100 == 0: print('.', end='')
                    a.phototransition('inactive_ground',
                                      'inactive_excited',
                                      intensity=0.01, # holding this constant
                                      polarization_xyz=(1, 0, 0))
                    a.phototransition('active_ground',
                                      'active_excited',
                                      intensity=sat,
                                      polarization_xyz=(0, 1, 0))
                    if i % 50==0 and animating and i!=0: 
                        # get the emission
                        x, y, z, t = a.get_xyzt_at_transitions('active_excited',
                                                               'active_ground')
                        # only add counts since we last plotted
                        idx = t > (i*10)-500
                        x_ = x[idx]; y_ = y[idx]; z_ = z[idx]
                        p_x, p_y = x_**2, y_**2 # Probabilities: channel x or y
                        r = uniform(0, 1, size=len(x_))
                        in_channel_x = (r < p_x)
                        in_channel_y = (p_x <= r) & (r < p_x + p_y)
                        # plot location of all emitters detected in X
                        chx_x = x_[in_channel_x]
                        chx_y = y_[in_channel_x]
                        chx_z = z_[in_channel_x]
                        ax.scatter(chx_x, chx_y, chx_z, s=15, c='#be03fd',
                                   marker="_", alpha=0.02)
                        # plot location of all emitters detected in Y
                        chy_x = x_[in_channel_y]
                        chy_y = y_[in_channel_y]
                        chy_z = z_[in_channel_y]
                        ax.scatter(chy_x, chy_y, chy_z, s=15, c='#ffa500',
                                   marker="|", alpha=0.02)
                        # plot the location & state of all current molecules
                        xg, yg, zg = a.get_xyz_for_state('active_ground')
                        molg = ax.scatter(xg, yg, zg, s=10, c='#808080')
                        xe, ye, ze = a.get_xyz_for_state('active_excited')
                        mole = ax.scatter(xe, ye, ze, s=10, c='#39ff14') 
                        # track the summed counts in each channel for the CDF
                        x, y = get_xy_emission_counts(a, 'active_excited',
                                                      'active_ground',
                                                      start_time_ns=(i*10)-500)
                        x_cts += x; y_cts += y
                        x_cdf.append(x_cts); y_cdf.append(y_cts)
                        output_path = out_dir/(
                            out_name+'_frame{:04d}.png'.format(int(i/50)))
                        fig.savefig(output_path, bbox_inches='tight', dpi=100)
                        molg.remove(); mole.remove()
                        time_list.append(i*10)
                    a.time_evolve(10)
                # Get the total summed counts
                x_total, y_total = get_xy_emission_counts(a, 'active_excited',
                                                          'active_ground')
                df = pd.DataFrame({'diffusion_time_ns': dtime,
                                   'saturation': sat,
                                   'counts_x': x_total,
                                   'counts_y': y_total,
                                   'n_fluorophores': n_fluor,
                                   'replicate': rep}, index=[0])
                results_list.append(df)
                if animating:
                    x_cdf.append(x_total); y_cdf.append(y_total)
                    time_list.append(100000)
                    df1 = pd.DataFrame({'time_ns': time_list,
                                        'x_cdf': x_cdf,
                                        'y_cdf': y_cdf,
                                        'diffusion_time_ns': dtime,
                                        'saturation': sat,
                                        'replicate': rep,
                                        'n_fluorophores': n_fluor},
                                       index=range(len(x_cdf)))
                    cdf_list.append(df1)
                    # save a final frame without any dots on top
                    output_path = out_dir/(out_name+'_frame0200.png')
                    fig.savefig(output_path, bbox_inches='tight', dpi=100)                    
results = pd.concat(results_list, ignore_index=True)
results.to_csv('photoswitching_summed_counts_1e2.csv')
cdfs = pd.concat(cdf_list, ignore_index=True)
cdfs.to_csv('photoswitching_cdfs_1e2.csv')
                           
