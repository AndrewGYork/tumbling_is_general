from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
import pandas as pd
from fluorophore_rotational_diffusion import Fluorophores, FluorophoreStateInfo

# This script simulates how a 'cap selected' population like you would
# produce with linearly polarized excitation light decays over time for
# different sized objects.
#
# It models the 'excited state' as infinitely long lived, so the key
# point to communicate here is the fundamentals of rotational diffusion
# without any photophysics layered on top.

def make_animation_frame(my_fluorophores, output_name, output_dir, frame_number,
                         cmap='inferno', view_angle=(90, -90), state=None,
                         label=None):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(1, 1, 1, projection='3d')
    if state == None: # default, plot all of the molecules
        o = my_fluorophores.orientations # nickname
        x = o.x; y=o.y; z=o.z
    else: # only plot a subset of the molecules
        x, y, z = my_fluorophores.get_xyz_for_state(state)
    ax.scatter(x, y, z, s=2,
               c=np.linspace(0, 1, len(x)),
               cmap=cmap, vmin=0, vmax=1)    
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
    if label is not None: plt.suptitle(label)
    output_path = output_dir/(output_name + '_frame{:04d}.png'.format(frame_number))
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

# Beginning of the simulation code
print("Generic Decay of an Oriented Population from a Cap Selection")
output_dir = Path(__file__).parents[4]/'tumbling_temp'/'cap_relaxation'/'individ'
output_dir.mkdir(exist_ok=True, parents=True)
diff2diam = {21746: 40, 73394: 60, 339789: 100, 2718318: 200}
diff2color = {21746: '#0072b2', 73394: '#009e73', 339789: '#e69f00',
              2718318: '#cc79a7'}
# Set up a custom colormap for the points for each size that will be
# intensity variations of the line color in the main plot
# https://matplotlib.org/stable/gallery/color/custom_cmap.html
cdict40 = {'red': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
           'green': ((0.0, 0.254, 0.254), (1.0, 0.639, 0.639)),
           'blue': ((0.0, 0.4, 0.4), (1.0, 1.0, 1.0))}
cdict60 = {'red': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
           'green': ((0.0, 0.322, 0.322), (1.0, 0.922, 0.922)),
           'blue': ((0.0, 0.231, 0.231), (1.0, 0.671, 0.671))}
cdict100 = {'red': ((0.0, 0.604, 0.604), (1.0, 1.0, 1.0)),
            'green': ((0.0, 0.416, 0.416), (1.0, 0.753, 0.753)),
            'blue': ((0.0, 0.0, 0.0), (1.0, 0.201, 0.201))}
cdict200 = {'red': ((0.0, 0.706, 0.706), (1.0, 0.882, 0.882)),
            'green': ((0.0, 0.271, 0.271), (1.0, 0.690, 0.690)),
            'blue': ((0.0, 0.510, 0.510), (1.0, 0.796, 0.796))}
diff2cdict = {21746: cdict40, 73394: cdict60, 339789: cdict100, 2718318: cdict200}

state_info = FluorophoreStateInfo()
state_info.add('ground')
state_info.add('excited') # pretend excited state is infinitely long lived
output_dfs = []
for diff in diff2diam.keys():
    print('\nSetting up fluorophores object for Diff. Time', diff)
    a = Fluorophores(1e6,
                     diffusion_time=diff,
                     state_info=state_info)
    a.phototransition('ground', 'excited', intensity=0.03,
                      polarization_xyz=(0, 1, 0))
    a.delete_fluorophores_in_state('ground')
    x, y, z = a.get_xyz_for_state('excited')
    x_sq = np.mean(x**2); y_sq = np.mean(y**2)
    x_sq_list = [x_sq]; y_sq_list = [y_sq]
    time_ns_list = [0]
    my_cmap = lsc('my_cmap', diff2cdict[diff])
    make_animation_frame(a, 'diff'+str(diff), output_dir, 0, cmap=my_cmap,
                         state='excited')
    # we'll only go for 40 us so the viewer doesn't get too bored
    for i in range(200):
        time_step_ns = 200
        a.time_evolve(time_step_ns)
        x, y, z = a.get_xyz_for_state('excited')
        x_sq = np.mean(x**2); y_sq = np.mean(y**2)
        x_sq_list.append(x_sq); y_sq_list.append(y_sq)
        time_ns_list.append((i+1) * time_step_ns)
        make_animation_frame(a, 'diff'+str(diff), output_dir, i+1, cmap=my_cmap,
                             state='excited')
        print('.', end='')
    assert len(x_sq_list) == len(y_sq_list) == len(time_ns_list)
    result = pd.DataFrame({'time_ns': time_ns_list,
                           'avg_x_squared': x_sq_list,
                           'avg_y_squared': y_sq_list,
                           'diff_time_ns': diff},
                          index = range(len(x_sq_list)))
    output_dfs.append(result)
results = pd.concat(output_dfs, ignore_index=True)
results.to_csv('01_cap_alignment_relaxation.csv')
        
plt.figure()
for name, group in results.groupby('diff_time_ns'):
    plt.plot(group['time_ns'], group['avg_y_squared'],
             color=diff2color[name], label=name)
plt.legend()
plt.savefig('02_cap_alignment_relaxation.png')
plt.show()
