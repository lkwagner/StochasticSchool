import numpy as np

# These Routines Help Measure the Energy for the Interacting and Non-Interacting Cases

#----------------------------------------------------------------------------
def compute_density_matrix(wf,trial_wf,size,n):
    # Routine to Determine the Up and Down One-Body Density Matrices

    # Compute Overlap Matrix transpose(PsiT)*Psi
    transpose_trial = trial_wf.transpose()

    # Get Overlap Matrix 
    overlap = transpose_trial.dot(wf)

    # Now Inverse Matrices
    inverse_overlap = np.linalg.inv(overlap)

    # Find Product Inverse Overlap and Trial WF Transpose
    inverse_trial_product = inverse_overlap.dot(trial_wf.transpose())

    # Multiple By Wf to Get Final One Body Density Matrix
    one_body_density_matrix = wf.dot(inverse_trial_product)

    return one_body_density_matrix

#----------------------------------------------------------------------------

def measure_onebody_energy(wf_up,wf_down,trial_wf_up,trial_wf_down,neighbors,size,n_up,n_down,t):
    # Measures the Non-Interacting Mixed Energy
    
    kinetic_energy = 0
 
    # Compute the One-Body Density Matrix
    one_body_density_matrix_up = compute_density_matrix(wf_up,trial_wf_up,size,n_up)
    one_body_density_matrix_down = compute_density_matrix(wf_down,trial_wf_down,size,n_down)
    
    # Compute Non-Interacting Energy Using Density Matrices
    for sites in range(size):

      if(size > 2):
        for j in range(2):
          kinetic_energy += one_body_density_matrix_up[sites,neighbors[sites,j]] + one_body_density_matrix_down[sites,neighbors[sites,j]]
      else: 
        for j in range(2):
          kinetic_energy += .5 * (one_body_density_matrix_up[sites,neighbors[sites,j]] + one_body_density_matrix_down[sites,neighbors[sites,j]])


    # Mulitply Kinetic Energy By Hopping Constant
    kinetic_energy *= -t
   
    return kinetic_energy

#----------------------------------------------------------------------------

def measure_total_energy(wf_up,wf_down,trial_wf_up,trial_wf_down,neighbors,size,n_up,n_down,U,t):
    # Measures the Non-Interacting Mixed Energy

    kinetic_energy = 0
    potential_energy = 0

    # Compute the One-Body Density Matrix
    one_body_density_matrix_up = compute_density_matrix(wf_up,trial_wf_up,size,n_up)
    one_body_density_matrix_down = compute_density_matrix(wf_down,trial_wf_down,size,n_down)

    # Compute Non-Interacting Energy Using Density Matrices
    for sites in range(size):

      potential_energy += one_body_density_matrix_up[sites,sites] * one_body_density_matrix_down[sites,sites]

      if(size > 2):
        for j in range(2):
          kinetic_energy += one_body_density_matrix_up[sites,neighbors[sites,j]] + one_body_density_matrix_down[sites,neighbors[sites,j]]
      else:
        for j in range(2):
          kinetic_energy += .5 * (one_body_density_matrix_up[sites,neighbors[sites,j]] + one_body_density_matrix_down[sites,neighbors[sites,j]])


    # Mulitply Kinetic Energy By Hopping Constant
    kinetic_energy *= -t
    potential_energy *= U
    total_energy = kinetic_energy + potential_energy

    return total_energy
  
#----------------------------------------------------------------------------
