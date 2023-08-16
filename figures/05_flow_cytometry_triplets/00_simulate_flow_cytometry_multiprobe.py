import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from fluorophore_rotational_diffusion import Fluorophores, FluorophoreStateInfo
from numpy.random import uniform
from math import sin, cos

# This code simulates a sequence of pulses on a flow cytometer at
# increasing delays. The modeled photophysics here is triplet
# triggering. This simulation differs from the camera-based simulation
# in that (1) the pulse durations are longer (some of them as long as
# 500 ns to model a cell flowing through a beam) and (2) the cell flows
# through each sequence <1 ms after the previous, so the fluorophore
# distribution carries over from sequence to sequence. (More concretely,
# we model all of the delays sequentially on one fluorophores object
# with a given rotational correlation time.)

def get_xy_emission_counts(population, initial_state, final_state,
                           start_time_ns=0):
    x, y, z, t = population.get_xyzt_at_transitions(initial_state, final_state)
    # to only look at counts from the most recent event, set a start time > 0
    trip_idx = t > start_time_ns
    x_t = x[trip_idx]; y_t = y[trip_idx]
    p_x, p_y = x_t**2, y_t**2 # Probabilities of landing in channel x or y
    r = uniform(0, 1, size=len(x_t))
    in_channel_x = (r < p_x)
    in_channel_y = (p_x <= r) & (r < p_x + p_y)
    return sum(in_channel_x), sum(in_channel_y)

# Some path setup
current_dir = Path(__file__).parents[0]

# Experimental parameters
diffusion_times = [450, 900, 3000]
delay_list = [100, 200, 400, 800, 1600, 3200, 6400] # matches camera-based
laser_separation_ns = 10000 # interval between probe laser and next pump
nrep = 12

# Let's begin the simulations
print("Simulation: Flow Cytometry with Triplets (Multi-pump, multi-probe)")
results_list = []
for dtime in diffusion_times:
    for rep in range(nrep):
        print('\n\nDiffusion time (ns) %0.1e, Replicate %d' % (dtime, rep))
        # Set up photophysics
        state_info = FluorophoreStateInfo()
        state_info.add('ground')
        state_info.add('excited_singlet', lifetime=2,
                       final_states=['ground', 'excited_triplet'],
                       probabilities=[0.99, 0.01])
        state_info.add('excited_triplet', lifetime=1e6,
                       final_states=['ground'])
        a = Fluorophores(1e6,
                         diffusion_time=dtime,
                         state_info=state_info)
        x_list = []; y_list = []
        time = 0
        for i, delay in enumerate(delay_list):
            print('Pump %d' % (i+1), 'Duration 50  ns')
            for x in range(10): # pump pulse, 50 ns duration           
                a.phototransition('ground', 'excited_singlet',
                                  intensity=2, # saturate to generate many singlets
                                  polarization_xyz=(0, 1, 0))
                a.time_evolve(5); time+=5
                print('.', end='')
            print('\nTime evolving delay of %d ns' % delay)
            a.time_evolve(delay); time+=delay
            print('Probe %d' % (i+1))
            # Circularly polarized probe (or our approximation to
            # circular polarization). Circular is preferable to linear
            # here because it will clean up more of the triplets before
            # the next pulse.
            for o in ((1, 0, 0), (0, 1, 0), (1, 1, 0), (1, -1, 0),
                      (sin(np.pi/8), cos(np.pi/8), 0),
                      (sin(3*np.pi/8), cos(3*np.pi/8), 0),
                      (sin(13*np.pi/8), cos(13*np.pi/8), 0),
                      (sin(15*np.pi/8), cos(15*np.pi/8), 0)):
                a.phototransition('excited_triplet', 'excited_singlet',
                                  intensity=1,
                                  polarization_xyz=o)
            a.time_evolve(50); time+=50 # let emissions occur
            # get any emissions since the start of the last probe pulse
            x, y = get_xy_emission_counts(a, 'excited_singlet', 'ground',
                                          start_time_ns=time-50)
            x_list.append(x); y_list.append(y) 
            # Move the cell down the flow cell to the next laser stage
            if i < (len(delay_list) - 1):
                print('\nMoving to the next laser')
                a.time_evolve(laser_separation_ns); time+=laser_separation_ns
        # record the output
        df = pd.DataFrame({'diffusion_time_ns': dtime,
                           'delay_ns': delay_list,
                           'counts_x': x_list,
                           'counts_y': y_list,
                           'replicate': rep}, index=range(len(x_list)))
        results_list.append(df)
results = pd.concat(results_list, ignore_index=True)
results.to_csv('flow_cytometry_circ_probe_12reps.csv')
                           

