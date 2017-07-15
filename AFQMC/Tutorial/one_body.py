# Module Containing One-Body Functions for Basic Manipulations

import numpy as np
import scipy as sci

#----------------------------------------------------------------------------
def neighbors_periodic_boundary_conditions(neighbors,size):
   # Form a Nearest Neighbors Table in neighbors

   # Run Through Sites Recording Names for Eachc
   #for sites in range(size):

     # Determine Neighboring Sites Assuming PBCs in One D
   pass

#----------------------------------------------------------------------------
def form_one_body_matrix(one_body_matrix,neighbors,size,t=-1):
   # Forms a One-Body Matrix for the Hubbard Model

   # Get Neighbors List

   # Create the One Body Density Matrix Based Upon the Neighbors List
   pass
   
#----------------------------------------------------------------------------
def exponentiate_one_body(one_body_propagator,one_body_matrix,trial_wf_up,trial_wf_down,size,n_up,n_down,delta_tau=.01):
   #Determines the Eigenvectors of the One Body Matrix and Exponentiates It

   # Determine Eigenvalues and Eigenvectors

   # Sort the Eigenvalues in Ascending (Increasing) Order

   # Store Eigenvectors of One Body Matrix As Trial WFs

   # Perform Matrix Exponentiation - Probably a Faster Expm Routine
   pass

#----------------------------------------------------------------------------
def propagate_one_body(one_body_propagator,wf_up,wf_down,size,n_up,n_down):
   # Propagates the Wave Function By Multiplying By Propagator

   # Obtain Temporary Wave Function From Product of Propagator and Current

   # Copy Over Temporary to Current
   pass
