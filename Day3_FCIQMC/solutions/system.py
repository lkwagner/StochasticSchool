import numpy as np
import math

class PARAMS:
    def __init__(self, totwalkers=10000, initwalkers = 10, init_shift=0.0, shift_damp=0.1, timestep=1.e-3,
           det_thresh=0.25, eqm_iters=50, max_iter=100000, stats_cycle=10, seed=7, init_thresh=None):
        ''' Class to set up fixed parameters for FCIQMC simulation'''

        # Set random number seed
        print('Setting random number seed to {}'.format(seed))
        np.random.seed(seed)

        # The initial number of walkers
        self.nwalk_init = float(initwalkers)

        # The target number of walkers in the calculation (it will grow to this) 
        self.nwalk_target = float(totwalkers)

        # Below sets a flag to determine whether we are in a fixed shift (growth phase) or variable shift (fixed walker number) phase
        if self.nwalk_init >= self.nwalk_target:
            self.fixedshift = False
            # Save the iteration number when we enter variable shift mode
            self.shift_vary_iter = 0 
        else:
            self.fixedshift = True
            # Save the iteration number when we enter variable shift mode
            self.shift_vary_iter = None

        # Store the shift value.
        self.shift = init_shift
        if self.shift < 0.0:
            print("Warning: Initial shift is < 0. Walkers growth may be negative?")
        # Once in variable shift, how damped do we want the fluctuations to be?
        self.shift_damp = 0.1

        # The timestep for the propagation
        self.timestep = timestep

        # The minimum value for an occupied determinant before it is reweighted/removed
        self.det_thresh = det_thresh

        # The number of equilibration iterations after entering variable shift 
        # mode before statistics are accumulated
        self.eqm_iters = eqm_iters

        # The total number of iterations to run for
        self.max_iter = max_iter

        # How often to write out intermediately accumulated statistics and update the shift. 'A' in notes.
        # We call this the 'update cycle' or 'block'
        self.stats_cycle = stats_cycle

        # The threshold for the initiator approximation (None = off)
        self.init_thresh = init_thresh
        return
        
    def update_shift(self, sim_stats):
        ''' Update the value of the shift, in order to stabilize the number of walkers over an update cycle. If we are in fixed shift mode, then test whether we want to come out of it.'''

        if self.fixedshift:
            if sim_stats.nw >= self.nwalk_target:
                print('Entering variable shift mode to stabilize walker number...')
                self.fixedshift = False
                self.shift_vary_iter = sim_stats.iter_curr

        if not self.fixedshift:
            # Update the value of the shift in order to stabilize the walker number between consecutive update cycles
            self.shift -= (self.shift_damp / (self.stats_cycle * self.timestep)) * \
                np.log(sim_stats.nw / sim_stats.nw_prev)
        return

class STATS:
    def __init__(self, sim_params, filename='fciqmc_stats', ref_energy=0.0):
        ''' Class for the accumulation of run-time variables, output 
        and methods required for computing the energy'''

        # The energy of the reference determinant (which shifts the propagator)
        self.ref_energy = ref_energy
        print('Reference energy shift in propagator: {}'.format(self.ref_energy))

        # The current number of walkers, and the number of walkers from the previous update cycle / block of stats
        self.nw = 0.0
        self.nw_prev = 0.0

        # Current iteration
        self.iter_curr = 0

        # Current weight on the reference determinant
        self.ref_weight = 0.0

        # Current number of occupied determinants
        self.nocc_dets = 0

        # This variable accumulates the numerator of the projected 
        # energy over the current iteration update cycle / block 
        self.cycle_en_num = 0.0
        # This variable accumulates the denominator of the projected 
        # energy over the current iteration update cycle / block
        self.cycle_en_denom = 0.0
        
        # List of stored energy numerator and denominator values of each block
        # over the accumulation iterations
        self.accum_en_num = []
        self.accum_en_denom = []

        # Open a file to output the statistics, and write out a header
        self.fout = open(filename,'w')
        self.fout.write('# 1.Iter    2.Shift   3.Walkers   4.Cycle_energy   5.Av.Energy  6.Error  7.#Occ_dets  8.Ref_weight\n')
        print('Iter    Shift   Walkers   Cycle_energy   Av.Energy  Error  #Occ_dets  Ref_weight')

        # Flag to indicate whether we are accumulating statistics yet or not
        if not sim_params.fixedshift and sim_params.eqm_iters == 0:
            self.stats_accum = True 
        else:
            self.stats_accum = False

        return

    def cycle_energy(self):
        ''' Return the energy averaged over the current cycle '''
        cycle_energy = self.cycle_en_num / self.cycle_en_denom + self.ref_energy
        return cycle_energy

    def av_energy(self):
        ''' Return the 'best' energy, averaged over all accumulation iterations.
            This computes the average and standard error of the numerator and 
            denominator of projected energy estimate, and propagates the errors.'''

        if len(self.accum_en_num) > 0:
            mean_num = np.mean(self.accum_en_num)
            mean_denom = np.mean(self.accum_en_denom)
            std_dev_num = np.std(self.accum_en_num)
            std_dev_denom = np.std(self.accum_en_denom)
            err_num = std_dev_num / math.sqrt(float(len(self.accum_en_num)))
            err_denom = std_dev_denom / math.sqrt(float(len(self.accum_en_denom)))

            # Find energy as average of numerator and denominator.
            # Why is this approach not ideal?
            av_energy = (mean_num / mean_denom) + self.ref_energy
            energy_err = abs(mean_num / mean_denom) * \
                math.sqrt((err_num/abs(mean_num))**2 + (err_denom/abs(mean_denom))**2)
        else:
            av_energy = 'n/a'
            energy_err = 'n/a'

        return av_energy, energy_err

    def update_stats(self, sim_params):
        ''' Function to accumulate the desired statistics'''

        # Do we want to start accumulating statistics?
        if sim_params.shift_vary_iter is not None and \
            self.iter_curr > sim_params.shift_vary_iter+sim_params.eqm_iters and \
            not self.stats_accum:
            print("Starting to accumulate averaged energy statistics...")
            self.stats_accum = True
        
        if self.stats_accum:
            # If we are accumulating statistics, take the stats over the update cycle, and store them in a list.
            self.accum_en_num.append(self.cycle_en_num / float(sim_params.stats_cycle))
            self.accum_en_denom.append(self.cycle_en_denom / float(sim_params.stats_cycle))

        # Change the previous update cycle number of walkers to be the current number
        # (Needed for the update of the shift)
        self.nw_prev = self.nw

        # Find the averaged energy (if accumulating)
        av_e, av_e_err = self.av_energy()

        # Write out the statistics to file, and stdout
        self.fout.write('{}   {}   {}   {}   {}   {}   {}   {}\n'.format(self.iter_curr, sim_params.shift, self.nw, self.cycle_energy(), av_e, av_e_err, self.nocc_dets, self.ref_weight)) 
        self.fout.flush()
        print('{}   {}   {}   {}   {}   {}   {}   {}'.format(self.iter_curr, sim_params.shift, self.nw, self.cycle_energy(), av_e, av_e_err, self.nocc_dets, self.ref_weight)) 
        
        # Re-zero cycle energy estimators for the next update cycle
        self.cycle_en_num = 0.0
        self.cycle_en_denom = 0.0

        return
