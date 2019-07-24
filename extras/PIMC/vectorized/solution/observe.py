import numpy as np
def draw_beads_3d(ax,beads):
  """ draw all beads in 3D
  Inputs:
   ax: matplotlib.Axes3D object
   beads: 3D numpy array of shape (nslice,nptcl,ndim)
  Output:
   ptcls: a list of pairs of plot objects. There is ony entry for each particle. Each entry has two items: line representing the particle and text labeling the particle.
  Effect:
   draw all particles on ax """

  nslice,nptcl,ndim = beads.shape
  com = beads.mean(axis=0) # center of mass of each particle, used to label the particles only

  ptcls = []
  for iptcl in range(nptcl):
    mypos = beads[:,iptcl,:] # all time slices for particle iptcl
    pos = np.insert(mypos,0,mypos[-1],axis=0) # close beads

    line = ax.plot(pos[:,0],pos[:,1],pos[:,2],marker='o') # draw particle
    text = ax.text(com[iptcl,0],com[iptcl,1],com[iptcl,2],'ptcl %d' % iptcl,fontsize=20) # label particle
    ptcls.append( (line,text) )
  return ptcls

def thermodynamic_kinetic(paths,lam,tau):
  """ thermodynmic estimator of the kinetic energy """
  nslice,nptcl,ndim,nconf = paths.shape
  ke = ndim*nptcl/2./tau * np.ones(nconf)
  for islice in range(nslice):
    r2_arr = (paths[islice]-paths[(islice+1)%nslice])**2. # (nptcl,ndim,nconf)
    r2 = r2_arr.sum(axis=0).sum(axis=0) # (nconf,)
    ke -= r2/(4.*lam*tau**2.)/nslice
  return ke
