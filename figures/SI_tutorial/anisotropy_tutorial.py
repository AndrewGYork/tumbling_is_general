from numpy.random import uniform
import numpy as np
import matplotlib.pyplot as plt
from fluorophore_rotational_diffusion import Fluorophores, FluorophoreStateInfo

# Set up the fluorophores object and run the simulation
tumbling_time_ns = 90
fluorescence_lifetime_ns = 2
state_info = FluorophoreStateInfo()
state_info.add('ground') # molecules default to being in the first state you specify
state_info.add('excited', lifetime=fluorescence_lifetime_ns,
               final_states='ground')
a = Fluorophores(number_of_molecules=5e7,
                 diffusion_time=tumbling_time_ns,
                 state_info=state_info)

a.phototransition('ground', 'excited',
                  intensity=0.01, polarization_xyz=(1, 0, 0))
for i in range(10):
    print('.', end='')
    a.delete_fluorophores_in_state('ground')
    a.time_evolve(fluorescence_lifetime_ns)

# Calculate steady-state anisotropy
x, y, z, t = a.get_xyzt_at_transitions('excited', 'ground')
p_x, p_y = x**2, y**2 # Probabilities of landing in channel x or y
r = uniform(0, 1, size=len(t))
in_channel_x = (r < p_x)
in_channel_y = (p_x <= r) & (r < p_x + p_y)
total_x = sum(in_channel_x)
total_y = sum(in_channel_y)
polarization = (total_x - total_y) / (total_x + total_y)
print('')
print('Total x (parallel) counts:', total_x)
print('Total y (perpendicular) counts:', total_y)
print('Polarization: {:0.2f}'.format(polarization))
# Sample output from this code (your results will vary slightly)

# Calculate time-resolved anisotropy
t_x, t_y = t[in_channel_x], t[in_channel_y]
bins = np.linspace(0, 5*fluorescence_lifetime_ns, 200)
bin_centers = (bins[1:] + bins[:-1])/2
(hist_x, _), (hist_y, _) = np.histogram(t_x, bins),  np.histogram(t_y, bins)

print("Saving results in test_classic_anisotropy_decay.png...", end='')
plt.figure()
plt.plot(bin_centers, hist_x, '.-', label=r'$\parallel$ polarization')
plt.plot(bin_centers, hist_y, '.-', label=r'$\perp$ polarization')
plt.title(
    "Simulation of classic time-resolved anisotropy decay\n" +
    r"$\tau_f$=%0.1f, $\tau_d$=%0.1f"%(a.state_info['excited'].lifetime,
                                       a.orientations.diffusion_time))
plt.xlabel("Time (ns)")
plt.ylabel("Photons per time bin")
plt.legend(); plt.grid('on')
plt.savefig("test_classic_anisotropy_decay.png"); plt.close()
print("done.")
# Result will be a file on disk with the graph similar to the graph below.
