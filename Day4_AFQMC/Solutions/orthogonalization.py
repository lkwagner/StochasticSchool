import numpy as np

# Orthogonalization of Wave Function Using Modified Gram Schmidt

#----------------------------------------------------------------------------
def orthogonalize(wf_up,wf_down,size,n_up,n_down):
   # Orthogonalize Wave Function to Prevent Collapse of Columns

   # Perform Modified Gram Schmidt on Each Wf
   wf_up = modified_gram_schmidt(wf_up,size,n_up)
   wf_down = modified_gram_schmidt(wf_down,size,n_down)

#----------------------------------------------------------------------------
def modified_gram_schmidt(q,size1,size2):
   # Orthogonalizes Columns of Matrices
  
   # Initialize Diagonal and R Pieces
   d = np.zeros(size2)
   r = np.zeros( (size2,size2) )
   temporary = 0.0
   anorm = 0.0

   # Perform Modified Gram Schmidt on Each Column - Could Be Faster Numpy Way
   for i in range(1,size2+1):
   
     temporary = 0.0
     for j in range(1,size1+1):
      temporary += q[j-1,i-1] * q[j-1,i-1]
     d[i-1]=np.sqrt(temporary)
     anorm = 1.0/d[i-1]

     for j in range(1,size1+1):
      q[j-1,i-1] *= anorm

     for j in range(i+1,size2+1):
      temporary = 0.0
      for k in range(1,size1+1):
         temporary += q[k-1,i-1] * q[k-1,j-1]
     
      for k in range(1,size1+1):
          q[k-1,j-1] -= (temporary * q[k-1,i-1])
      r[i-1,j-1]=temporary*anorm
  
   for i in range(size2):
     r[i,i] = d[i]

   return q

#----------------------------------------------------------------------------
