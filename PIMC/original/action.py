import numpy as np
def HarmonicOscillator(pos,mass=1.0,omega=1.0):
  
  # --- solution ---- 
  r2  = (pos*pos).sum() # sum over particles and dimensions
  pot = 0.5*mass*omega**2.*r2
  # --- solution ---- 
  return pot
