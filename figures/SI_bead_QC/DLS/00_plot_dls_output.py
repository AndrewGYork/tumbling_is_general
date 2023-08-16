from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size

# This script plots the result of size distributions generated within
# the NanoTemper DLS software.

# Read in the outputs for each size (from the NanoTemper software)
d2 = Path(__file__).parents[4] / 'tumbling_temp' / 'source_data' / '2023-03-29_DLS'
plt.style.use(Path(__file__).parents[2] / 'default.mplstyle')
d = d2 / 'Export_2023-04-24_13-26-18'
diam_list = [40, 60, 100, 200]
df_list = []
for diam in diam_list:
    df_ = pd.read_csv(d / str(diam) / (str(diam) + '.csv'))
    df_['bead_diam_nm'] = diam
    df_list.append(df_)
df = pd.concat(df_list, ignore_index=True)
df.to_csv('01_combined_output.csv')

diam2color = {40: '#0072b2', 60: '#009e73', 100: '#e69f00', 200: '#cc79a7'}

# Make a plot of the bead samples
fig1 = plt.figure(figsize=(4,4))
h = [Size.Fixed(1.0), Size.Fixed(2)]
v = [Size.Fixed(0.7), Size.Fixed(2)]
divider = Divider(fig1, (0, 0, 1, 1), h, v, aspect=False)
axs1 = fig1.add_axes(divider.get_position(),
                     axes_locator=divider.new_locator(nx=1, ny=1))
for name, group in df.groupby('bead_diam_nm'):
    color = diam2color[name]
    plt.semilogx(group['Radius [nm]'], group['Relative frequency [%]'],
                 label=str(round(name/2))+' nm', color=color)
    plt.fill_between(group['Radius [nm]'],
                     group['Relative frequency [%]'] - group['S.D.'],
                     group['Relative frequency [%]'] + group['S.D.'],
                     alpha=0.3, color=color)
axs1.set_xlim(5, 10000)
axs1.set_xlabel('Estimated Radius (nm)')
axs1.set_ylabel('Relative Frequency (%)')
axs1.legend(title='Theor. Radius')
axs1.grid()
plt.savefig('02_relative_frequency_distributions.png', bbox_inches='tight')

plt.show()
