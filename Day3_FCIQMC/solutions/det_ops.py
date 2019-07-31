import numpy as np

class HAM:
    def __init__(self, filename = 'FCIDUMP', p_single = 0.05):
        ''' Define a hamiltonian to sample, as well as its quantum numbers.
        In addition, it defines the probability of generating a single excitation, rather than double
        excitation in the random excitation generator (which is a method of this class).
        Finally, it also defines a reference determinant energy.'''

        # All these quantities are defined by the 'read_in_fcidump' method
        self.nelec = None       # Number of electrons
        self.ms = None          # 2 x spin-polarization
        self.n_alpha = None     # Number of alpha electrons
        self.n_beta = None      # Number of beta electrons
        self.nbasis = None      # Number of spatial orbitals
        self.spin_basis = None  # Number of spin orbitals (= 2 x self.nbasis)

        # The one-electron hamiltonian in the spin-orbital basis 
        self.h1 = None
        # The two-electron hamiltonian in the spin-orbital basis
        # Note that eri[i,j,k,l] = < phi_i(r_1) phi_k(r_2) | 1/r12 | phi_j(r_1) phi_l(r_2) >
        # This ordering is called 'chemical ordering', and means that the first two indices of the array
        # define the charge density for electron 1, and the second two for electron two.
        self.h2 = None
        # The (scalar) nuclear-nuclear repulsion energy
        self.nn = None

        self.read_in_fcidump(filename)

        # The probability of generating a single excitation rather than a double excitation
        self.p_single = p_single

        # Define a reference determinant and reference energy.
        # Ideally, this should be the energy of the lowest-energy determinant in the space.
        # In a HF basis, this will generally be the first occupied orbitals. Assume so.
        self.ref_det = list(range(self.n_alpha)) + list(range(self.nbasis, self.nbasis+self.n_beta))
        print('Initial reference determinant defined with occupied orbitals: {}'.format(self.ref_det))
        return

    def read_in_fcidump(self, filename):
        ''' This function looks (and is!) pretty messy. Don't worry about it too much. It sets up the system parameters
        as defined by the hamiltonian in the file. It sets the following system parameters:

        self.nelec          # Number of electrons
        self.ms             # 2 x spin-polarization
        self.n_alpha        # Number of alpha electrons
        self.n_beta         # Number of beta electrons
        self.nbasis         # Number of spatial orbitals
        self.spin_basis     # Number of spin orbitals (= 2 x self.nbasis)

        as well as the integrals defining the hamiltonian terms:
        self.h1[:,:]        # A self.spin_basis x self.spin_basis matrix of one-electron terms
        self.h2[:,:,:,:]    # A rank-4 self.spin_basis array for the two electron terms
        self.nn             # The (scalar) nuclear repulsion energy

        Note that the integrals are defined in the spin-orbital basis, and the self.h2 term is defined as follows:
        eri[i,j,k,l] = < phi_i(r_1) phi_k(r_2) | 1/r12 | phi_j(r_1) phi_l(r_2) >
        This ordering is called 'chemical ordering', and means that the first two indices of the array
        define the charge density for electron 1, and the second two for electron two.'''
        import os
        import re

        print('Reading in system from file: {}'.format(filename))
        assert(os.path.isfile(os.path.join('./', filename)))

        finp = open(filename, 'r')
        dat = re.split('[=,]', finp.readline())
        while not 'FCI' in dat[0].upper():
            dat = re.split('[=,]', finp.readline())
        self.nbasis = int(dat[1])
        print('Number of spatial orbitals in the system: {}'.format(self.nbasis))
        self.nelec = int(dat[3])
        print('Number of electrons in the system: {}'.format(self.nelec))
        self.ms = int(dat[5])
        print('2 x Spin polarization of system: {}'.format(self.ms))
        self.n_alpha = (self.ms + self.nelec) // 2
        self.n_beta = self.nelec - self.n_alpha
        print('Number of (alpha, beta) electrons: {}, {}'.format(self.n_alpha, self.n_beta))

        # Read in symmetry information, but we are not using it
        sym = []
        dat = finp.readline().strip()
        while not 'END' in dat:
            sym.append(dat)
            dat = finp.readline().strip()

        isym = [x.split('=')[1] for x in sym if 'ISYM' in x]
        if len(isym) > 0:
            isym_out = int(isym[0].replace(',','').strip())
        symorb = ','.join([x for x in sym if 'ISYM' not in x]).split('=')[1]
        orbsym = [int(x.strip()) for x in symorb.replace(',', ' ').split()]

        # Read in integrals, but immediately transform them into a spin-orbital basis.
        # We order things with alpha, then beta spins
        self.spin_basis = 2*self.nbasis
        self.h1 = np.zeros((self.spin_basis, self.spin_basis))
        # Ignore permutational symmetry
        self.h2 = np.zeros((self.spin_basis, self.spin_basis, self.spin_basis, self.spin_basis))
        dat = finp.readline().split()
        while dat:
            ii, jj, kk, ll = [int(x) for x in dat[1:5]] # Note these are 1-indexed
            i = ii-1
            j = jj-1
            k = kk-1
            l = ll-1
            if kk != 0:
                # Two electron integral - 8 spatial permutations x 4 spin (=32) allowed permutations!
                # alpha, alpha, alpha, alpha
                self.h2[i, j, k, l] = float(dat[0])
                self.h2[j, i, k, l] = float(dat[0])
                self.h2[i, j, l, k] = float(dat[0])
                self.h2[j, i, l, k] = float(dat[0])
                self.h2[k, l, i, j] = float(dat[0])
                self.h2[l, k, i, j] = float(dat[0])
                self.h2[k, l, j, i] = float(dat[0])
                self.h2[l, k, j, i] = float(dat[0])

                # beta, beta, beta, beta
                self.h2[i+self.nbasis, j+self.nbasis, k+self.nbasis, l+self.nbasis] = float(dat[0])
                self.h2[j+self.nbasis, i+self.nbasis, k+self.nbasis, l+self.nbasis] = float(dat[0])
                self.h2[i+self.nbasis, j+self.nbasis, l+self.nbasis, k+self.nbasis] = float(dat[0])
                self.h2[j+self.nbasis, i+self.nbasis, l+self.nbasis, k+self.nbasis] = float(dat[0])
                self.h2[k+self.nbasis, l+self.nbasis, i+self.nbasis, j+self.nbasis] = float(dat[0])
                self.h2[l+self.nbasis, k+self.nbasis, i+self.nbasis, j+self.nbasis] = float(dat[0])
                self.h2[k+self.nbasis, l+self.nbasis, j+self.nbasis, i+self.nbasis] = float(dat[0])
                self.h2[l+self.nbasis, k+self.nbasis, j+self.nbasis, i+self.nbasis] = float(dat[0])

                # alpha, alpha, beta, beta
                self.h2[i, j, k+self.nbasis, l+self.nbasis] = float(dat[0])
                self.h2[j, i, k+self.nbasis, l+self.nbasis] = float(dat[0])
                self.h2[i, j, l+self.nbasis, k+self.nbasis] = float(dat[0])
                self.h2[j, i, l+self.nbasis, k+self.nbasis] = float(dat[0])
                self.h2[k, l, i+self.nbasis, j+self.nbasis] = float(dat[0])
                self.h2[l, k, i+self.nbasis, j+self.nbasis] = float(dat[0])
                self.h2[k, l, j+self.nbasis, i+self.nbasis] = float(dat[0])
                self.h2[l, k, j+self.nbasis, i+self.nbasis] = float(dat[0])

                # beta, beta, alpha, alpha
                self.h2[i+self.nbasis, j+self.nbasis, k, l] = float(dat[0])
                self.h2[j+self.nbasis, i+self.nbasis, k, l] = float(dat[0])
                self.h2[i+self.nbasis, j+self.nbasis, l, k] = float(dat[0])
                self.h2[j+self.nbasis, i+self.nbasis, l, k] = float(dat[0])
                self.h2[k+self.nbasis, l+self.nbasis, i, j] = float(dat[0])
                self.h2[l+self.nbasis, k+self.nbasis, i, j] = float(dat[0])
                self.h2[k+self.nbasis, l+self.nbasis, j, i] = float(dat[0])
                self.h2[l+self.nbasis, k+self.nbasis, j, i] = float(dat[0])
            elif kk == 0:
                if jj != 0:
                    # One electron term
                    self.h1[i,j] = float(dat[0])
                    self.h1[j,i] = float(dat[0])
                    self.h1[i+self.nbasis, j+self.nbasis] = float(dat[0])
                    self.h1[j+self.nbasis, i+self.nbasis] = float(dat[0])
                else:
                    # Nuclear repulsion term
                    self.nn = float(dat[0])
            dat = finp.readline().split()

        print('System file read in.')
        finp.close()
        return

    def slater_condon(self, det, excited_det, excit_mat, parity):
        ''' Calculate the hamiltonian matrix element between two determinants, det and excited_det.
        In:
            det:            A list of occupied orbitals in the original det (note should be ordered)
            excited_det:    A list of occupied orbitals in the excited det (note should be ordered)
            excit_mat:      A list of two tuples, giving the orbitals excited from and to respectively
                                (i.e. [(3, 6), (0, 12)] means we have excited from orbitals 3 and 6 to orbitals 0 and 12
                                    for a single excitation, the tuples will just be of length 1)
                                Note: For a diagonal matrix element (i.e. det == excited_det), the excit_mat should be 'None'.
            parity:         The parity of the excitation
        Out: 
            The hamiltonian matrix element'''

        hel = 0.0
        if excit_mat is None:
            # Diagonal Hamiltonian matrix element
            assert(det == excited_det)

            # Include nuclear-nuclear repulsion
            hel += self.nn

            for i in range(self.nelec):
                # Sum over all diagonal terms in one-electron operator
                hel += self.h1[det[i],det[i]]
                for j in range(i+1, self.nelec):
                    # Run through electron pairs and sum in 'coulomb' and 'exchange' contribution
                    hel += ( self.h2[det[i],det[i],det[j],det[j]] - self.h2[det[i],det[j],det[j],det[i]] )
        elif len(excit_mat[0]) == 1:
            # Single excitation

            # One electron part to single excitation
            hel += self.h1[excit_mat[0][0], excit_mat[1][0]]
            # Two electron part to single excitation
            for i in det:
                hel += ( self.h2[excit_mat[0][0], excit_mat[1][0], i, i] - self.h2[excit_mat[0][0], i, i, excit_mat[1][0]] )

            # Multiply by the parity of the excitation
            hel *= parity
        elif len(excit_mat[0]) == 2:
            # Double excitation

            # Just a single 'coulomb'-like and 'exchange'-like contribution
            hel += ( self.h2[excit_mat[0][0], excit_mat[1][0], excit_mat[0][1], excit_mat[1][1]] - \
                        self.h2[excit_mat[0][0], excit_mat[1][1], excit_mat[0][1], excit_mat[1][0]] )
            # Multiply by the parity of the excitation
            hel *= parity
        return hel

    def excit_gen(self, det):
        ''' Take in a determinant, and create a single or double excitation or it.
        This does *not* take into account any spin (or spatial) symmetries.
        The determinant is represented as an ordered list of occupied orbital indices.
        Returns:
            o The singly or doubly-excited determinant as an ordered orbital list (self.p_single should determine this probability)
            o The excitation matrix giving the orbital indices which change (see docstring in the slater_condon function above for definition)
            o The parity of the excitation
            o The normalized probability of the excitation'''

        # Create an equivalent unoccupied orbital list, by removing the occupied elements
        unocc_list = [x for x in range(self.spin_basis) if x not in det]

        if np.random.rand(1)[0] < self.p_single:
            # Generate a single excitation
            # Pick a single occupied orbital and unoccupied orbital index
            i_ind = np.random.choice(range(len(det)), 1)[0]
            a_ind = np.random.choice(range(len(unocc_list)), 1)[0]

            # Find the overall probability of generating this excitation
            prob = self.p_single / float(self.nelec * (self.spin_basis - self.nelec))
            excited_det = det[:]    # Make copy
            # Make the replacement in the orbital for the new determinant
            excited_det[i_ind] = unocc_list[a_ind]

            # Find the parity of the excitation by finding out how to order many electron swaps are required to order the resulting orbital string
            perm = elec_exchange_ops(excited_det, i_ind)
            # This hasn't sorted the excited determinant, so do this now
            excited_det.sort()
            
            # Create a pair of tuples for the annihilated and created orbital
            excit_mat = [(det[i_ind],), (unocc_list[a_ind],)]

        else:
            # Generate a double excitation
            # Pick two orbitals from the occupied list, and two from the unoccupied list *without* replacement
            i_ind = tuple(np.random.choice(range(len(det)), 2, replace=False))
            a_ind = tuple(np.random.choice(range(len(unocc_list)), 2, replace=False))
            prob = 4. * (1. - self.p_single) / \
                float(self.nelec * (self.nelec-1) * (self.spin_basis - self.nelec) * (self.spin_basis - self.nelec - 1))
            # Replace one electron at a time and calculate permutation for each (which we can do, since we are only interested in odd/even). As long as we reorder after each replacement
            excited_det = det[:] # Make copy
            excited_det[i_ind[0]] = unocc_list[a_ind[0]]

            # Find the parity of the first replacement
            perm1 = elec_exchange_ops(excited_det, i_ind[0])
            # We have to resort the determinant to calculate the parity for replacement two
            excited_det.sort()
            # However, this reordering *may* have changed the index of the electron to replace in the determinant.
            # Refind the correct index
            ind = excited_det.index(det[i_ind[1]])
            excited_det[ind] = unocc_list[a_ind[1]]

            # Find the parity of the second replacement
            perm2 = elec_exchange_ops(excited_det, ind)
            perm = perm1 + perm2    # Find the total number of permutations
            # Sort the new determinant for the final time
            excited_det.sort()

            # Create a pair of tuples for the annihilated and created orbitals
            excit_mat = [(det[i_ind[0]], det[i_ind[1]]), (unocc_list[a_ind[0]], unocc_list[a_ind[1]])]

        return excited_det, excit_mat, (-1)**perm, prob

def elec_exchange_ops(det, ind):
    ''' Given a determinant defined by a list of occupied orbitals
    which is ordered apart from one element (ind), find the number of
    local (nearest neighbour) electron exchanges required to order the 
    list of occupied orbitals.
    
    We can assume that there are no repeated elements of the list, and that
    the list is ordered apart from one element on entry.
    
    Return: The number of pairwise permutations required.'''

    # A better way to do it!
    if True:

        a_orb = det[ind]
        det_sort = sorted(det)
        newind = det_sort.index(a_orb)
        perm = abs(newind - ind)
    else:
        # A brute force way to do it!

        n = len(det)    # The number of electrons
        perm = 0

        # Do we want to do pairwise permutations going up (i.e. element is too 
        # small for its position), or down (i.e. element is too large for 
        # its position) the list.
        search_up = True
        if ind == 0:
            # The replaced element is at the beginning of the list
            search_up = True
        elif ind == n-1:
            # The replaced element is at the end of the list
            search_up = False
        elif det[ind-1] > det[ind]:
            # The replaced element is smaller than the preceeding element.
            # We therefore have to search down (otherwise, up)
            search_up = False

        if search_up:
            for x in range(ind+1,n):
                if det[ind] < det[x]:
                    # We have gone through enough pair-wise permutations
                    # and have now found its rightful spot!
                    break
                perm += 1
        else:
            for x in range(ind-1,-1,-1):
                if det[ind] > det[x]:
                    # We have gone through enough pair-wise permutations
                    # and have now found its rightful spot!
                    break
                perm += 1
    return perm

def calc_excit_mat_parity(det, excited_det):
    ''' Given two determinants (excitations of each other), calculate and return 
    the excitation matrix (see the definition in the slater-condon function), 
    and parity of the excitation'''

    # First, we have to compute the indices which are different in the two orbital lists.
    excit_mat = []
    excit_mat.append(tuple(set(det) - set(excited_det)))    # These are the elements in det which are not in excited_det
    excit_mat.append(tuple(set(excited_det) - set(det)))    # These are the elements in excited_det which are not in det
    assert(len(excit_mat[0]) == len(excit_mat[1]))

    # Now find the parity
    new_det = det[:]
    perm = 0
    for elec in range(len(excit_mat[0])):

        # Find the index of the electron we want to remove
        ind_elec = new_det.index(excit_mat[0][elec])
        # Substitute the new electron
        new_det[ind_elec] = excit_mat[1][elec]
        # Find the permutation of this replacement
        perm += elec_exchange_ops(new_det, ind_elec)
        # Reorder the determinant
        new_det.sort()
    assert(new_det == excited_det)

    return excit_mat, (-1)**perm

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Test for hamiltonian matrix elements
    # Read in System 
    sys_ham = HAM(filename='FCIDUMP.8H')
    print('Running unit tests for Slater-Condon rules...')
    # Define set of excitations to compute the hamiltonian matrix element
    test_ham_els = [(sys_ham.ref_det, sys_ham.ref_det, None, None),
                    ([1, 4, 6, 7, 8, 9, 10, 11], [1, 4, 6, 7, 8, 9, 10, 11], None, None),
                    ([0, 1, 3, 4, 8, 9, 11, 12], [0, 2, 3, 4, 8, 9, 11, 12], [(1,), (2,)], 1),
                    ([0, 2, 3, 7, 9, 10, 11, 13], [0, 2, 3, 4, 7, 9, 10, 11], [(13,), (4,)], 1),
                    ([0, 2, 3, 7, 9, 10, 11, 13], [0, 2, 5, 7, 9, 11, 13, 15], [(10, 3), (5, 15)], -1),
                    ([0, 1, 2, 3, 8, 9, 10, 11], [0, 2, 3, 5, 8, 10, 11, 14], [(9, 1), (14, 5)], 1)]
    # The correct matrix elements
    correct_hels = [-4.000299230765899, -1.7706124224297999, 0.003968296598667837, 0.0,
                    -0.008689269052231, 0.0001549635629506746]
    # Test each one.
    for i, (det_i, det_j, excit_mat, parity) in enumerate(test_ham_els):
        hel = sys_ham.slater_condon(det_i, det_j, excit_mat, parity)
        if np.allclose(hel, correct_hels[i]):
            print('Hamiltonian matrix element correct! H element = {}'.format(hel))
        else:
            print('*Hamiltonian matrix element incorrect!*')
            print('Initial determinant: {}'.format(str(det_i)))
            print('Excited determinant: {}'.format(str(det_j)))
            print('Orbital excitation matrix: {}'.format(str(excit_mat)))
            print('Parity of excitation: {}'.format(parity))
            print('Expected hamiltonian matrix element: {}'.format(correct_hels[i]))
            print('Returned hamiltonian matrix element: {}'.format(hel))

    # elec_exchange_ops unit tests 
    print('Running unit tests for elec_exchange_ops function...')
    # A list of test determinants and exchanged orbital indices.
    test_dets = [([0,1,2,3,4,5], 0), 
                 ([0,1,2,3,4,5], 3),
                 ([0,1,2,3,4,5], 5),
                 ([3,6,14,8,11], 2),
                 ([14,3,6,8,11], 0),
                 ([0,8,3,6,11],  1),
                 ([1,3,12,0],    3),
                 ([1,3,0,12],    2),
                 ([1,3,2,4],     2),
                 ([1,3,2,4],     1)]
    correct_perms = [0, 0, 0, 2, 4, 2, 3, 2, 1, 1]
    for i, (det, ind) in enumerate(test_dets):
        perm = elec_exchange_ops(det, ind)
        if perm == correct_perms[i]:
            print('Permutation correct! Perm = {}'.format(perm))
        else:
            print('*Permutation incorrect*!')
            print('True permutation number for determinant {} is {}'.format(str(det),correct_perms[i]))
            print('Your function returns instead: {}'.format(perm))

    # excit_gen unit tests
    # Use random initial determinant value 
    det_root = [0, 1, 4, 6, 8, 12, 13, 15]
    print('Running unit tests for excitation generation function, from determinant {}...'.format(str(det_root)))
    sys_ham = HAM(filename='FCIDUMP.8H',p_single=0.1)
    n_att = 20000
    excited_dets = {}
    for attempt in range(n_att):
        if attempt % 1000 == 0:
            print('Generated {} excitations...'.format(attempt))
        # Generate a random excitation from this root determinant
        excited_det, excit_mat, parity, prob = sys_ham.excit_gen(det_root)

        # Check that the returned determinant is the same length as the original determinant
        assert(len(det_root) == len(excited_det))

        # Check that returned determinant is an ordered list
        assert(all(excited_det[i] <= excited_det[i+1] for i in range(len(excited_det)-1)))

        # Store the excited determinant
        if repr(excited_det) in excited_dets:
            excited_dets[repr(excited_det)] += 1./(prob*n_att)
        else:
            excited_dets[repr(excited_det)] = 1./(prob*n_att)
    # Create list of n_gen / (N_att x prob) for all excited determinants
    print('Total number of excitations generated: {}'.format(len(excited_dets)))
    probs = []
    for excited_det, prob_sum in list(excited_dets.items()):
        print('Excitation generated: {}'.format(excited_det))
        probs.append(prob_sum)
    plt.plot(range(len(probs)), probs, label='normalized generation frequency')
    plt.axhline(1.0,label='Exact distribution desired')
    plt.legend()
    plt.show()

    # Test example for parity
    det_root = [0, 1, 4, 6, 8, 12, 13, 15]
    print('Running unit tests for calc_excit_mat_parity function, from determinant {}...'.format(str(det_root)))
    sys_ham = HAM(filename='FCIDUMP.8H',p_single=0.1)
    # Generate a number of excitations
    for i in range(25):
        # Generate a random excitation from this root determinant
        excited_det, excit_mat, parity, prob = sys_ham.excit_gen(det_root)

        # Now check that the excit_mat and parity are the same if calculated
        # independently from the calc_excit_mat_parity function.
        excit_mat_2, parity_2 = calc_excit_mat_parity(det_root, excited_det)
        if excit_mat_2 == excit_mat and parity == parity_2:
            print('Excitation matrix and parity agree between the two functions for attempt {}'.format(i))
        # Note that the parity should change if we swap the indices of either the excited from or excited to orbitals
        elif parity == -parity_2 and [tuple(reversed(excit_mat[0])), excit_mat[1]] == excit_mat_2:
            print('Excitation matrix and parity agree between the two functions for attempt {} (though the "from" indices are swapped)'.format(i))
        elif parity == -parity_2 and [excit_mat[0], tuple(reversed(excit_mat[1]))] == excit_mat_2:
            print('Excitation matrix and parity agree between the two functions for attempt {} (though the "to" indices are swapped)'.format(i))
        elif parity == parity_2 and [tuple(reversed(excit_mat[0])), tuple(reversed(excit_mat[1]))] == excit_mat_2:
            print('Excitation matrix and parity agree between the two functions for attempt {} (though both sets of indices are swapped)'.format(i))
        else:
            print('Error in getting agreement for the excitation matrix and parity...')
            print('Root determinant: {}'.format(str(det_root)))
            print('Excited determinant: {}'.format(str(excited_det)))
            print('excit_gen parity = {}. calc_excit_mat_parity parity = {}'.format(parity, parity_2))
            print('excit_gen excitation matrix = {}. calc_excit_mat_parity excitation matrix = {}'.format(excit_mat, excit_mat_2))
