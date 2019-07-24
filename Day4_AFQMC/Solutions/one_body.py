# Module Containing One-Body Functions for Basic Manipulations

import numpy as np
import scipy as sci

#----------------------------------------------------------------------------
def neighbors_periodic_boundary_conditions(neighbors,size):
   # Form a Nearest Neighbors Table in neighbors

   # Run Through Sites Recording Names for Eachc
   for sites in range(size):

     # Determine Neighboring Sites Assuming PBCs in One D
     if(sites!=size-1 and sites!=0):
        neighbors[sites,0] = sites-1
        neighbors[sites,1] = sites+1
     elif(sites==size-1):
        neighbors[sites,0] = sites-1
        neighbors[sites,1] = 0
     elif(sites==0):
        neighbors[sites,0] = size-1
        neighbors[sites,1] = sites+1

#----------------------------------------------------------------------------
def form_one_body_matrix(one_body_matrix,neighbors,size,t=-1):
   # Forms a One-Body Matrix for the Hubbard Model

   # Get Neighbors List
   neighbors_periodic_boundary_conditions(neighbors,size) 

   # Create the One Body Density Matrix Based Upon the Neighbors List
   for sites in range(size):
    for adjacent in range(2):
      one_body_matrix[sites, neighbors[sites,adjacent]] = -t

#----------------------------------------------------------------------------
def exponentiate_one_body(one_body_propagator,one_body_matrix,trial_wf_up,trial_wf_down,size,n_up,n_down,delta_tau=.01):
   #Determines the Eigenvectors of the One Body Matrix and Exponentiates It

   # Determine Eigenvalues and Eigenvectors
   eigvals, eigvecs = np.linalg.eig(one_body_matrix)

   # Sort the Eigenvalues in Ascending (Increasing) Order
   idx = eigvals.argsort()[::1]
   eigvals = eigvals[idx]
   eigvecs = eigvecs[:,idx]

   # Store Eigenvectors of One Body Matrix As Trial WFs
   for i in range(int(n_up)):
     trial_wf_up[:,i] = eigvecs[:,i]

   for i in range(int(n_down)):
     trial_wf_down[:,i] = eigvecs[:,i]

   # Perform Matrix Exponentiation - Probably a Faster Expm Routine
   for k in range(size):
    for i in range(size):
     for j in range(size):
       one_body_propagator[i,j] += np.exp(-delta_tau/2.0 * eigvals[k])*eigvecs[i,k]*eigvecs[j,k]


#----------------------------------------------------------------------------
def propagate_one_body(one_body_propagator,wf_up,wf_down,size,n_up,n_down):
   # Propagates the Wave Function By Multiplying By Propagator

   # Obtain Temporary Wave Function From Product of Propagator and Current
   temp_wf_up = one_body_propagator.dot(wf_up)
   temp_wf_down = one_body_propagator.dot(wf_down)

   # Copy Over Temporary to Current
   wf_up = temp_wf_up
   wf_down = temp_wf_down 

