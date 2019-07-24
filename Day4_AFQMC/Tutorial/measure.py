import numpy as np

# These Routines Help Measure the Energy for the Interacting and Non-Interacting Cases

#----------------------------------------------------------------------------
def compute_density_matrix(wf,trial_wf,size,n):
    # Routine to Determine the Up and Down One-Body Density Matrices

    # Compute Overlap Matrix transpose(PsiT)*Psi

    # Get Overlap Matrix 

    # Now Inverse Matrices

    # Find Product Inverse Overlap and Trial WF Transpose

    # Multiple By Wf to Get Final One Body Density Matrix

    #return one_body_density_matrix
    pass

#----------------------------------------------------------------------------

def measure_onebody_energy(wf_up,wf_down,trial_wf_up,trial_wf_down,neighbors,size,n_up,n_down,t):
    # Measures the Non-Interacting Mixed Energy
    
    kinetic_energy = 0
 
    # Compute the One-Body Density Matrix
    #one_body_density_matrix_up = compute_density_matrix(wf_up,trial_wf_up,size,n_up)
    #one_body_density_matrix_down = compute_density_matrix(wf_down,trial_wf_down,size,n_down)
    
    # Compute Non-Interacting Energy Using Density Matrices

    # Mulitply Kinetic Energy By Hopping Constant
  
    pass 
    #return kinetic_energy

#----------------------------------------------------------------------------

def measure_total_energy(wf_up,wf_down,trial_wf_up,trial_wf_down,neighbors,size,n_up,n_down,U,t):
    # Measures the Non-Interacting Mixed Energy

    kinetic_energy = 0
    potential_energy = 0

    # Compute the One-Body Density Matrix
    one_body_density_matrix_up = compute_density_matrix(wf_up,trial_wf_up,size,n_up)
    one_body_density_matrix_down = compute_density_matrix(wf_down,trial_wf_down,size,n_down)

    # Compute Non-Interacting Energy Using Density Matrices

    # Mulitply Kinetic Energy By Hopping Constant

    pass
    #return total_energy
  
#----------------------------------------------------------------------------
