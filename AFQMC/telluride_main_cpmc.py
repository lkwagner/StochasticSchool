import numpy as np
import one_body as ob 
import two_body as tb
import measure as ms
import orthogonalization as og

def main():

    # Constants for Now - Should Read In
    t=1
    U=0.0
    size = 4
    delta_tau=.01
    n_up = 2
    n_down = 2
    total_steps = 10000
    steps_orthog = 10
    steps_measure = 20

    # Matrices Needed Throughout Simulation
    one_body_matrix = np.zeros( (size,size))
    one_body_propagator = np.zeros( (size,size))
    neighbors = np.zeros( (size,2) )
    trial_wf_up = np.zeros( (size, n_up))
    trial_wf_down = np.zeros( (size, n_down))
    wf_up = np.zeros( (size, n_up) )
    wf_down = np.zeros( (size, n_down) )

    # Form the One Body Portion of the Hamiltonian
    ob.form_one_body_matrix(one_body_matrix,neighbors,size,t)

    # Exponentiate That One Body Portion of the Hamiltonian So You Can Propagate
    ob.exponentiate_one_body(one_body_propagator,one_body_matrix,trial_wf_up,trial_wf_down,size,n_up,n_down,delta_tau)

    # Copy Trial Wave Function to Current Wave Function For Propagattion Loop
    wf_up = trial_wf_up
    wf_down = trial_wf_down

    # Propagate By Looping Over Products of One and Two Body Propagators
    for steps in range(total_steps): 

       # First Propagate by Half of One Body Term
       ob.propagate_one_body(one_body_propagator,trial_wf_up,trial_wf_down,size,n_up,n_down)

       # Then Propagate By Two Body Term
       tb.propagate_two_body(wf_up,wf_down,size,n_up,n_down,U,delta_tau)

       # Lastly Propagate by Half of One Body Term
       ob.propagate_one_body(one_body_propagator,trial_wf_up,trial_wf_down,size,n_up,n_down)

       # Orthogonalize the Wave Functions to Ensure No Collapse
       if(steps%steps_orthog == 0):
         og.orthogonalize(wf_up,wf_down,size,n_up,n_down) 

       # Measure Observables Like the Energy Every So Often
       if(steps%steps_measure == 0):
         current_energy = ms.measure_total_energy(wf_up,wf_down,trial_wf_up,trial_wf_down,neighbors,size,n_up,n_down,U,t)
       
         print current_energy
 
         # Append File with Values
         f = open("energy.dat", "a+")
         f.write("%i %5.2f\n" % (steps, current_energy))  

    f.close() 

if __name__ == "__main__":
   main()

