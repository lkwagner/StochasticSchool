#!/usr/bin/env python
import numpy as np

# ---- solution ----
def action_for_slice(path,islice):
  action = 0.0
  action += path.KineticAction(islice-1,islice) + path.KineticAction(islice,islice+1)
  action += path.PotentialAction(islice-1,islice) + path.PotentialAction(islice,islice+1)
  return action
# ---- solution ----

def SingleSliceMove(path,sigma=0.1):
  path.RelabelBeads() # pick a random time slice to call 0
  accepted = False

  # ---- solution ----
  move = sigma*np.random.randn(path.nptcl,path.ndim)

  islice = 1 # always move time slice 1


  # calculate current value of actions related to time slice 1
  cur_action_to_change = action_for_slice(path,islice)

  # move path
  path.beads[islice,:,:] += move

  # re-calculate action pieces
  action_after_change = action_for_slice(path,islice)

  action_change = action_after_change - cur_action_to_change

  # accept or reject
  prob = np.exp(-action_change)
  if np.random.rand()<prob:
    accepted = True
  else:
    path.beads[islice,:,:] -= move
  # ---- solution ----
  return accepted

def pimc(nstep,path,move_list,block_size=10):
  nblock  = nstep//block_size
  nmove   = len(move_list)

  nslice,nptcl,ndim = path.beads.shape
  path_trace = np.zeros([nblock,nslice,nptcl,ndim])
  etrace  = np.zeros(nblock) # energy trace
  naccept = np.zeros(nmove,dtype=int) # number of accepted moves, resolved for each move
  iblock = 0
  for istep in range(nstep):
    for imove in range(len(move_list)):
      if (move_list[imove](path)):
        naccept[imove] += 1
      if istep%block_size == 0:
        etrace[iblock] = path.Energy()
        path_trace[iblock] = path.beads.copy()
        iblock += 1
  return naccept,etrace,path_trace

def test_single_slice_move():
  from path import Path
  #from action import HarmonicOscillator
  #test_pot = lambda x:HarmonicOscillator(x,omega=10.)
  tau = 0.1
  lam = 0.5

  nslice = 5
  nptcl  = 2
  ndim   = 3
  path = Path(np.zeros([nslice,nptcl,ndim]),tau,lam)
  #path.SetPotential(test_pot)
  path.SetPotential(lambda x:0.0)
  path.SetCouplingConstant(0.0)

  nstep = 4000
  move_list = [SingleSliceMove]
  naccept,etrace,path_trace = pimc(nstep,path,move_list)
  nmove = len(move_list)*nstep
  acceptance_rate = float(naccept.sum())/nmove
  print('acceptance rate: {acc:4.2f}%'.format(acc=acceptance_rate*100.) )

  #np.savetxt('energy.dat',etrace)
  #np.savetxt('beads_trace.dat',path_trace.flatten())

  nequil = 50 # in blocks
  from CalcStatistics import Stats
  energy_mean,energy_error,energy_correlation =  Stats(etrace[nequil:])

  beta = tau*nslice
  energy_expect = 1.5*beta
  print(energy_mean,energy_error)
  print(energy_expect)

  # use 3 sigma tolerance
  assert abs(energy_mean-energy_expect) < 3*energy_error

if __name__ == '__main__':
  print('testing single bead move ...')
  test_single_slice_move()
  print ('single bead move passed')
