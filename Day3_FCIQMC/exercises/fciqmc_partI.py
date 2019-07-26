import numpy as np
import system

# Read in the full Hamiltonian of all configurations
full_h = np.load('Full_Ham_6H.npy')
ndet = full_h.shape[0]
print('Hamiltonian read in of dimension {} x {}'.format(ndet,ndet))

# Exactly diagonalize the full hamiltonian, to get a benchmark number for the exact ground state energy
print('Completely diagonalizing full hamiltonian (may be slow...)')
e, v = np.linalg.eigh(full_h)
print('Ground state energy of full hamiltonian is {}'.format(e[0]))

# Find the reference energy from the first element of the Hamiltonian.
# Define h_sim, which is where the reference energy has been removed from the diagonal of the hamiltonian
ref_energy = full_h[0,0]
h_sim = full_h - np.eye(full_h.shape[0])*ref_energy
print('Removing reference energy from simulated hamiltonian')

# Setup simulation parameters. See system.py for details.  
sim_params = system.PARAMS(totwalkers=2000, initwalkers=10, init_shift=0.0, 
        shift_damp=0.1, timestep=1.e-2, det_thresh=0.25, eqm_iters=250, 
        max_iter=15000, stats_cycle=10, seed=7)
# Setup a statistics object, which accumulates various run-time variables.
# See system.py for more details.
sim_stats = system.STATS(sim_params, filename='fciqmc_stats', ref_energy=ref_energy)

# Set up walker object as a dictionary.
# Initially, label determinants by their index in the hamiltonian array.
walkers = {0: sim_params.nwalk_init}
sim_stats.nw = sim_params.nwalk_init

# Loop through iterations
for sim_stats.iter_curr in range(sim_params.max_iter):

    spawned_walkers = {}    # Dictionary to hold the spawned walkers of each iteration
    sim_stats.nw = 0.0      # Recompute number of walkers each iteration
    sim_stats.ref_weight = 0.0
    sim_stats.nocc_dets = 0

    # Iterate over occupied determinants (keys) in the walker dictionary
    # Since we are modifying inplace, want to convert to a list, rather than setting up an iterator
    for det_ind, det_amp in list(walkers.items()):

        # Accumulate current walker contribution to energy expectation values
        # and update reference weight
        if det_ind == 0:
            sim_stats.cycle_en_denom += det_amp
            sim_stats.ref_weight = det_amp
        else:
            sim_stats.cycle_en_num += det_amp*h_sim[det_ind,0]

        # Stochastically round the walkers, if their amplitude is too low to ensure the walker list remains compact.
        # This will rely on determining whether the amplitude is above or below sim_params.det_thresh (\chi in notes), and either
        # stochastically increasing it to this threshold value, or deleting it and skipping to the next determinant.
        pass

        # Update statistic for the number of walkers, and number of occupied determinants
        sim_stats.nw += abs(det_amp)
        sim_stats.nocc_dets += 1

        # Do a number of SPAWNING STEPS proportional to the modulus of the determinant amplitude, and
        # put any spawned walkers in the appropriate dictionary.
        pass

        # DEATH STEP: Modify the amplitude of the determinant
        pass
        

    # ANNIHILATION. Run through the list of newly spawned walkers, and merge with the main list.
    pass
            
# Every sim_params.stats_cycle iterations, readjust shift (if in variable shift mode) and print out statistics.
    if sim_stats.iter_curr % sim_params.stats_cycle == 0:
        
        # Update the shift, and turn on variable shift mode if enough walkers
        sim_params.update_shift(sim_stats)
        # Update the averaged statistics, and write out 
        sim_stats.update_stats(sim_params)

# Close the output file
sim_stats.fout.close()
