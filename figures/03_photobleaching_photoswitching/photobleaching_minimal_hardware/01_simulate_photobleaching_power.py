import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
from pathlib import Path
import pandas as pd
from fluorophore_rotational_diffusion import Fluorophores, FluorophoreStateInfo
from numpy.random import uniform
from math import floor

# This script simulates the use of polarized photobleaching light to
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

def make_animation_frame(my_fluorophores, output_name, output_dir, frame_number,
                         states=None, cmaps=['#808080', '#39ff14'],
                         view_angle=(90, -90), label=None):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.computed_zorder=False
    if states == None: # default, plot all of the molecules
        o = my_fluorophores.orientations # nickname
        x = o.x; y=o.y; z=o.z
    else: # only plot a subset of the molecules
        for i, state in enumerate(states):
            x, y, z = my_fluorophores.get_xyz_for_state(state)
            # if we just use a linspace, lighter stuff will be plotted
            # last. in dense areas, this will look like the whole object
            # is lighter than it should be. As a workaround, we'll make
            # 'c' a bit more sophisticated.
            a = 1 if state == 'excited_singlet' else 0.4
            c = np.concatenate((np.linspace(0, 1, floor(len(x)/4)),
                                np.linspace(1, 0, floor(len(x)/4)),
                                np.linspace(0, 1, floor(len(x)/4)),
                                np.linspace(1, 0, (floor(len(x)/4) + len(x)%4))))
            ax.scatter(x, y, z, s=4, c=c, cmap=cmaps[i], vmin=0, vmax=1, alpha=a,
                       zorder=i+1)    
    ax.view_init(*view_angle)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_box_aspect((1, 1, 1))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticklabels([])
        axis._axinfo['juggled'] = (1, 1, 1)
        axis._axinfo['tick']['inward_factor'] = 0
        axis._axinfo['tick']['outward_factor'] = 0
        axis._axinfo['grid']['color'] = '#dcdcdc'
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    if label is not None: ax.text(0, -1.1, 0, label, ha='center', fontsize=20)
    output_path = output_dir/(output_name + '_frame{:04d}.png'.format(frame_number))
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

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
out_dir = cwd.parents[3] / 'tumbling_temp' / 'photobleaching' / 'dots_on_a_sphere'
out_dir.mkdir(exist_ok=True, parents=True)

# set up colormaps for the animation plotting
cdictG = {'red': ((0.0, 0.650, 0.650), (1.0, 0.349, 0.349)),
          'green': ((0.0, 0.650, 0.650), (1.0, 0.349, 0.349)),
          'blue': ((0.0, 0.650, 0.650), (1.0, 0.349, 0.349))}
graysG = lsc('graysG', cdictG)
cdictE = {'red': ((0.0, 0.118, 0.118), (1.0, 0.153, 0.153)),
          'green': ((0.0, 0.784, 0.784), (1.0, 0.984, 0.984)),
          'blue': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))}
greensE = lsc('graysG', cdictE)

# Conditions we will test
diffusion_times = [750, 1200, 3000, 6e3] # antibody + (monomer, small-med-large agg.)
saturations = [0.01, 0.05, 0.25, 1.25, 6.25]
nreps = 3

# Let's begin the simulations
print("Simulation: Photobleaching (Rotational FRAP)")
results_list = []; cdf_list = []
for dtime in diffusion_times:
    for sat in saturations:
        for rep in range(nreps):
            animating=False
            if sat == 1.25 and dtime in [750, 6e3] and rep == 0:
                animating=True
            print('\n\nDiff. time %0.1e, saturation %0.3f' % (dtime, sat))
            print('Replicate', rep)
            out_name = "photobleaching_dt%0.1e_sat%0.3f_%d"%(dtime, sat, rep)
            out_name = out_name.replace('.', 'p')
            # Set up photophysics
            state_info = FluorophoreStateInfo()
            state_info.add('ground')
            state_info.add('bleached')
            state_info.add('excited_singlet', lifetime=1,
                           final_states=['ground', 'bleached'],
                           probabilities=[0.95, 0.05])
            a = Fluorophores(1e4,
                             diffusion_time=dtime,
                             state_info=state_info)
            print('Photobleaching', end='')
            x_cts, y_cts = get_xy_emission_counts(a, 'excited_singlet', 'ground')
            if animating:
                x_cdf = []; y_cdf = []; time_list = []
            # Let's bleach out a stripe while recording counts
            for i in range(10000):
                a.phototransition('ground',
                                  'excited_singlet',
                                  intensity=sat,
                                  polarization_xyz=(0, 1, 0))
                if i % 100 == 0: print('.', end='')
                if i % 10 == 0:
                    x, y = get_xy_emission_counts(a, 'excited_singlet', 'ground',
                                                  start_time_ns = (i-10)*10)
                    x_cts += x; y_cts += y
                    if animating and i<1500:
                        make_animation_frame(a, out_name, out_dir, int(i/10),
                                             states=['excited_singlet', 'ground'],
                                             cmaps=[greensE, graysG])
                        x_cdf.append(x_cts); y_cdf.append(y_cts)
                        time_list.append(i*10)
                a.time_evolve(10) # should be pretty much fully decayed from singlet
                a.delete_fluorophores_in_state('bleached')
            x, y = get_xy_emission_counts(a, 'excited_singlet', 'ground',
                                          start_time_ns = 9990*10) # get last cycle
            x_cts += x; y_cts += y
            df = pd.DataFrame({'diffusion_time_ns': dtime,
                               'saturation': sat,
                               'counts_x': x_cts,
                               'counts_y': y_cts,
                               'replicate': rep}, index=[0])
            results_list.append(df)
            if animating:
                df1 = pd.DataFrame({'time_ns': time_list,
                                    'x_cdf': x_cdf,
                                    'y_cdf': y_cdf,
                                    'diffusion_time_ns': dtime,
                                    'saturation': sat,
                                    'replicate': rep},
                                   index=range(len(x_cdf)))
                cdf_list.append(df1)
results = pd.concat(results_list, ignore_index=True)
results.to_csv('photobleaching_summed_counts_1e4.csv')
cdfs = pd.concat(cdf_list, ignore_index=True)
cdfs.to_csv('photobleaching_cdfs_1e4.csv')
