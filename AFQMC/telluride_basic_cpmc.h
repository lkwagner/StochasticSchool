#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "/gpfs/runtime/opt/intel/2013.1.106/mkl/include/mkl.h"

/*****************************************************************************************************************/

#define IM1          2147483563
#define AM           (1.0 / (double)IM1)
#define IMM1         (IM1 - 1)
#define NTAB         32
#define NDIV         (1 + IMM1 / NTAB)
#define REPS         1.2e-7
#define RNMX         (1.0 - REPS)
#define IA 16807
#define IM 2147483647
#define IQ 127773
#define IR 2836
#define EPS 1.2e-20
#define RADIX 2.0
#define TINY 1.0e-20

/*******************************************************************************************************/

typedef struct st1 {
  double U, t; /*Extended Hubbard Model Parameters*/
  double delta_tau; /*Time Slice Size*/
  double exponential_constant; /*This Is The Constant Used in the Exponent for CPMC*/
} cns_st;

typedef struct st2 {
  int n_up;     /*Number Up Electrons*/
  int n_down;   /*Number Down Electrons*/
  int n_sites;  /*Number of Sites - Assuming Linear*/
  int n_sites_sq; /*Square of the Number of Sites To Simplify Expressions*/
  int n_steps;  /*Number of CPMC Steps*/
} int_st;


/*Obtain Parameters from Files*/
void get_parameters(int_st *ist,cns_st *cns); 

/*One-Body Functions*/
void form_one_body(double *one_body_matrix,int *neighbors,int_st ist,cns_st cns);
void neighbors_periodic_boundary_conditions(int *neighbors, int_st ist); 
void get_trial_wave_functions(double *trial_wf_up,double *trial_wf_down,double *non_interacting_eigenvalues,double *non_interacting_eigenvectors,int_st ist); 
void exponentiate_one_body(double *one_body_propagator,double *one_body_matrix,double *trial_wf_up,double *trial_wf_down,int_st ist,cns_st cns); 
void propagate_one_body(double *one_body_propagator,double *wf_up,double *wf_down,int_st ist); 

/*Two-Body Function*/
void propagate_two_body(double *wf_up,double *wf_down,int_st ist,cns_st cns); 

/*Orthogonalization*/
void modified_gram_schmidt(double *q, double *r, int size1, int size2); 
void orthogonalize(double *wf_up,double *wf_down,int_st ist); 

/*Compute Energy*/
double measure_onebody_energy(double *wf_up, double *wf_down, double *trial_wf_up, double *trial_wf_down,int *neighbors, int_st ist, cns_st cns); 
double measure_total_energy(double *wf_up, double *wf_down, double *trial_wf_up, double *trial_wf_down,int *neighbors, int_st ist, cns_st cns);
void compute_density_matrix(double *wf_up,double *wf_down,double *trial_wf_up,double *trial_wf_down,double *one_body_density_matrix_up,double *one_body_density_matrix_down,int_st ist); 
void inverse(double *matrix, double *inverse, int size); 
void inverse_2(double *matrix, double *inverse, int size); 
void lubksb(double *matrix, int n, int *indx, double *b); 
void ludmp(double *matrix, int n, int *indx, double *d); 

/*Random Number Generators*/
float gasdev(long *idum); 
float ran1(long *idum); 
void Randomize(); 

/*Zeros Vectors*/
void dzero(double *vec, int size);

/*Copies Vectors*/
void dcopy_2(double *new_vec, double *old_vec, int size);   
