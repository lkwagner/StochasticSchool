import numpy as np

# Contains All Two-Body Matrix Routines

#----------------------------------------------------------------------------
def propagate_two_body(wf_up,wf_down,size,n_up,n_down,U=0.0, delta_tau=0.1):
   # Propagates By Two Body Terms As Separated by HS Transform

   # Form One Body Constant In Front of HS Transformation
   exponential_onebody = np.exp(-delta_tau * U/2.0)
  
   # Form Constant That Goes Into Two-Body Exponential
   exponential_constant = np.sqrt( U * delta_tau ) 

   # Run Through Sites Sampling Fields and Forming One Body HS Terms*/
   for sites in range(size):

    # Generate Gaussian-Distributed Random Field   
    field = np.random.normal()

    # Store Exponential Two Body Terms
    exponential_twobody_up = np.exp(field * exponential_constant)
    exponential_twobody_down = np.exp(-1 * field * exponential_constant)

    # Multiply Constants Into Wave Functions
    for i in range(int(n_up)):
      wf_up[sites,i] *= (exponential_onebody * exponential_twobody_up)
    
    for i in range(int(n_down)):
      wf_down[sites,i] *= (exponential_onebody * exponential_twobody_down)

#----------------------------------------------------------------------------

