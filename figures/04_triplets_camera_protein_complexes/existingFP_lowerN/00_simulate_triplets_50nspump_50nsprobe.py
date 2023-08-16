from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
from numpy.random import uniform
import pandas as pd
from fluorophore_rotational_diffusion import Fluorophores, FluorophoreStateInfo

# This script simulates the use of a pump probe pulse scheme (i.e.
# camera system) to measure protein interactions in cells. We track how
# the polarization ratio of a protein tagged with an FP differs over
# time if it is alone, part of a small protein complex, or part of a
# larger protein complex.

# Although the underlying simulation code is in principle unitless, it
# is convenient to assume that 1 time unit is equivalent to 1 ns.

# This script runs 9 replicates of each condition. These replicates can
# be used to estimate noise or to model mixtures by adding different
# replicates together. It models a fluorophore with an ideal triplet
# yield (50%) and many molecules to extract a "true" picture of the
# polarization decay.

def get_xy_emission_counts(population, initial_state, final_state, time_cutoff=0):
    x, y, z, t = population.get_xyzt_at_transitions(initial_state, final_state)
    # threshold based on time before summing to remove singlets
    trip_idx = t > time_cutoff
    x_t = x[trip_idx]; y_t = y[trip_idx]
    p_x, p_y = x_t**2, y_t**2 # Probabilities of landing in channel x or y
    r = uniform(0, 1, size=len(x_t))
    in_channel_x = (r < p_x)
    in_channel_y = (p_x <= r) & (r < p_x + p_y)
    return sum(in_channel_x), sum(in_channel_y)

print("Pump Probe: Triplets with Small-ish Proteins")
current = Path(__file__).parents[0]
diffusion_times_ns = [450, 900, 3000]
probe_delays_ns = [100, 200, 400, 800, 1600, 3200, 6400]
nrep = 9 # for later summing and making mixtures
n_fluorophores = 1e5
triplet_qy = 0.01

state_info = FluorophoreStateInfo()
state_info.add('ground')
state_info.add('excited_singlet', lifetime=2,
               final_states=['ground', 'excited_triplet'],
               probabilities=[(1-triplet_qy), triplet_qy])
state_info.add('excited_triplet',
               lifetime=1e6, # ~1 ms triplet lifetime for current FPs
               final_states='ground')
output_dfs = []
for diff in diffusion_times_ns:
    for delay in probe_delays_ns:
        for rep in range(nrep):
            print('\nDiff. Time', diff, '& Probe Delay', delay, '(ns)')
            a = Fluorophores(n_fluorophores,
                             diffusion_time=diff,
                             state_info=state_info)
            # Leave the excitation laser on for a few singlet lifetimes
            print('Generating triplet population', end='')
            for x in range(10): # simulate a 50 ns pump "pulse"           
                a.phototransition('ground', 'excited_singlet', intensity=0.25,
                                  polarization_xyz=(0, 1, 0))
                a.time_evolve(5)
                print('.', end='')
            print('\nTime Evolving and Triggering')
            a.delete_fluorophores_in_state('ground') # performance
            a.time_evolve(delay)
            for x in range(10): # simulate a 50 ns probe "pulse"
                a.phototransition('excited_triplet', 'excited_singlet',
                                  intensity=0.25, polarization_xyz=(1, 0, 0))
                a.time_evolve(5)
            a.time_evolve(50)# allow newly generated singlets enough time to emit
            x, y = get_xy_emission_counts(a, 'excited_singlet', 'ground',
                                          time_cutoff=150) # 50 ns illum, 50 ns decay
            result = pd.DataFrame({'x': x,
                                   'y': y,
                                   'diff_time_ns': diff,
                                   'delay_time_ns': delay,
                                   'replicate': rep,
                                   'n_fluorophores': n_fluorophores,
                                   'triplet_QY': triplet_qy},
                                  index = [0])
            output_dfs.append(result)
results = pd.concat(output_dfs, ignore_index=True)
name = '01_pump_probe_n{:0.1e}_tqy{:0.1f}_{:d}reps_2pulses'.format(
    n_fluorophores, triplet_qy, nrep)
name = name.replace('.', 'p')
results.to_csv(name+'.csv')
