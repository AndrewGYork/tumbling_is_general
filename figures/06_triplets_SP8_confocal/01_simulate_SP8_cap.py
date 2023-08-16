import numpy as np
from pathlib import Path
import pandas as pd
from fluorophore_rotational_diffusion import Fluorophores, FluorophoreStateInfo
from numpy.random import uniform

# This script simulates the triplets pump probe experiment run on the
# commercially available SP8 confocal microscope. It simulates a pump
# and probe pair with a given delay; delays span from 60 to 900 us. To
# more closely  match the parameters on the SP8, which used pulsed
# lasers, the pump and probe are each represented as 16 pulses (matching
# the 200 ns dwell time and 80 MHz repetition rate on the instrument).

# The fluorophore parameters (fluorescence lifetime and triplet yield)
# were chosen to correspond to mVenus, but they are sufficiently close
# the values for mScarlet that these results can be extended to that
# case as well.

# The sample simulated here is fluorescently labeled beads of 4
# different diameters (40, 60, 100, and 200 nm). A key difference
# between the simulation and the actual sample is that we are simulating
# "single" spherical fluorophores of a given size, but the real samples
# were spheres covered with many individual fluorophores. This change
# will affect the noise properties but should not confound the general
# mean polarization ratio for a sufficiently large number of fluorophores.

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

current_dir = Path.cwd()
diffusion_times_ns = [21746, 73394, 339789, 2718318] # 40, 60, 100, 200 nm diameter
delay_list_us = list(range(60, 960, 60))
nrep = 12
results_list = []

for dtime in diffusion_times_ns:
    for delay_us in delay_list_us:
        for rep in range(nrep):
            print('\n\nTumbling time (ns) %0.1e, Delay (us) %d, Replicate %d' % (
                dtime, delay_us, rep))
            # Set up photophysics
            state_info = FluorophoreStateInfo()
            state_info.add('ground')
            state_info.add('excited_singlet', lifetime=2, # mVenus
                           final_states=['ground', 'excited_triplet'],
                           probabilities=[0.99, 0.01])
            state_info.add('excited_triplet', lifetime=1e6, # mVenus
                           final_states=['ground'])
            a = Fluorophores(1e6,
                             diffusion_time=dtime,
                             state_info=state_info)
            print('Generating initial singlet population')
            for x in range(16): # 200 ns dwell at 80 MHz rep rate
                a.phototransition('ground', 'excited_singlet',
                                  intensity=2, polarization_xyz=(1, 0, 0))
                a.time_evolve(12.5)
            a.delete_fluorophores_in_state('ground') # performance
            print('Waiting for Probe...')
            a.time_evolve(delay_us*1000)
            print('Triggering triplets')
            for x in range(16): # again, 200 ns dwell at 80 MHz rep rate
                a.phototransition('excited_triplet', 'excited_singlet',
                                  intensity=0.25, polarization_xyz=(0, 1, 0))
                a.time_evolve(12.5)
            x, y = get_xy_emission_counts(a, 'excited_singlet', 'ground',
                                          start_time_ns=500)
            # record the output
            df = pd.DataFrame({'diffusion_time_ns': dtime,
                               'delay_us': delay_us,
                               'counts_x': x,
                               'counts_y': y,
                               'replicate': rep}, index=[0])
            results_list.append(df)
results = pd.concat(results_list, ignore_index=True)
results.to_csv('sp8_simulation_pump2_probe0p25.csv')
