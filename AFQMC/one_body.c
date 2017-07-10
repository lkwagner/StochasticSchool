/*This Contains All One-Body Matrix Formation and Propagation Routines*/

#include "telluride_basic_cpmc.h"

/*************************************************************************************************/

void form_one_body(double *one_body_matrix,int *neighbors,int_st ist, cns_st cns){

   /*Forms the One-Body Matrix for the Hubbard Model*/
   int j, sites; 

   /*Zero One-Body Matrix*/
   dzero(one_body_matrix,ist.n_sites_sq);  

   /*First Obtain the Nearest Neighbors of All Sites Under PBCS*/
   neighbors_periodic_boundary_conditions(neighbors,ist); 

   /*Create One-Body Matrix Based Upon Neighbors List*/
   for (sites=0; sites<ist.n_sites; sites++) {

     /*Run Through All Neighbors of Sites - Only Two If Linear*/
     for (j=0; j<2; j++) {
      one_body_matrix[sites*ist.n_sites+neighbors[sites*2+j]]=-cns.t; 
     }
   }

return; 
}

/*************************************************************************************************/

void exponentiate_one_body(double *one_body_propagator,double *one_body_matrix,double *trial_wf_up,double *trial_wf_down,int_st ist,cns_st cns){

   /*Forms the Exponential of the One-Body Matrix By Diagonalizing; Also Stores Non-Interacting Trial Wave Functions*/
   int info; 
   int i, j, k;
   double *non_interacting_eigenvalues, *non_interacting_eigenvectors; 

   /*Initialize Eigenvalues and Eigenvectors Matrices*/
   non_interacting_eigenvalues = (double *)malloc(ist.n_sites*sizeof(double)); 
   non_interacting_eigenvectors = (double *)malloc(ist.n_sites_sq*sizeof(double));  

   /*Zero One-Body Propagator and Wave Functions*/
   dzero(one_body_propagator,ist.n_sites_sq);
   dzero(non_interacting_eigenvalues,ist.n_sites);
   dzero(non_interacting_eigenvectors,ist.n_sites_sq);    
   dzero(trial_wf_up,ist.n_sites*ist.n_up); 
   dzero(trial_wf_down,ist.n_sites*ist.n_down); 

   /*Copy One Body Matrix Into Non-Interacting Eigenvectors - Replace with Blas*/
   dcopy_2(non_interacting_eigenvectors,one_body_matrix,ist.n_sites_sq); 

   /*Diagonalize the One-Body Matrix*/ 
   /*Now Diagonalize Both Matrices*/
    info = LAPACKE_dsyev(LAPACK_ROW_MAJOR,'V','U',ist.n_sites,non_interacting_eigenvectors,ist.n_sites,non_interacting_eigenvalues);

   /*Construct Exponential of One-Body*/
   for (k=0; k<ist.n_sites; k++) {
     for (i=0; i<ist.n_sites; i++) {
      for (j=0; j<ist.n_sites; j++) { /*Note That These Propagate A Half-Step Forward*/
        one_body_propagator[i*ist.n_sites+j] +=exp(-cns.delta_tau/2.0 * non_interacting_eigenvalues[k])*non_interacting_eigenvectors[i*ist.n_sites+k]*non_interacting_eigenvectors[j*ist.n_sites+k];
      }
     }
    }
 
    /*Obtain Non-Interacting = Trial Wave Functions from Eigenvalues and Eigenvectors*/
    get_trial_wave_functions(trial_wf_up,trial_wf_down,non_interacting_eigenvalues,non_interacting_eigenvectors,ist); 

free(non_interacting_eigenvalues); 
free(non_interacting_eigenvectors); 
return; 
}

/**********************************************************************************************/

void get_trial_wave_functions(double *trial_wf_up,double *trial_wf_down,double *non_interacting_eigenvalues,double *non_interacting_eigenvectors,int_st ist) {   

   int i, j, k;
   double p;  

   /*Given the Eigenvalues and Eigenvectors of the One-Body Matrix, Get Trial Wave Functions*/
   for (i=1;i<ist.n_sites;i++) {
     k=i;
     p=non_interacting_eigenvalues[k-1];
     for (j=i+1;j<=ist.n_sites;j++) {
       if (non_interacting_eigenvalues[j-1] >= p) {k=j; p=non_interacting_eigenvalues[k-1]; };
     }
     if (k != i) {
        non_interacting_eigenvalues[k-1]=non_interacting_eigenvalues[i-1];
        non_interacting_eigenvalues[i-1]=p;
        for (j=1;j<=ist.n_sites;j++) {
           p=non_interacting_eigenvectors[(j-1)*ist.n_sites+i-1];
           non_interacting_eigenvectors[(j-1)*ist.n_sites+i-1]=non_interacting_eigenvectors[(j-1)*ist.n_sites+k-1];
           non_interacting_eigenvectors[(j-1)*ist.n_sites+k-1]=p;
         }
      }
    }
  
   /*Copy Lowest Up Down Eigenvectors Into Matrices*/
   for (i=0; i<ist.n_up; i++) {
      for (j=0; j<ist.n_sites; j++) {
        trial_wf_up[j*ist.n_up+i] = non_interacting_eigenvectors[j*ist.n_sites+(ist.n_sites-i-1)];
      }
   }

   for (i=0; i<ist.n_down; i++) {
      for (j=0; j<ist.n_sites; j++) {
        trial_wf_down[j*ist.n_down+i] = non_interacting_eigenvectors[j*ist.n_sites+(ist.n_sites-i-1)];
      }
   }

return; 
}

/*************************************************************************************************/

void propagate_one_body(double *one_body_propagator,double *wf_up,double *wf_down,int_st ist) {

   /*Propagate the Wave Function by the One-Body Operator*/
   double *temp_wf_up, *temp_wf_down;  

   temp_wf_up = (double *)malloc(ist.n_sites*ist.n_up*sizeof(double)); 
   temp_wf_down = (double *)malloc(ist.n_sites*ist.n_down*sizeof(double));

   /*Zero New Wave Function*/
   dzero(temp_wf_up,ist.n_sites*ist.n_up); 
   dzero(temp_wf_down,ist.n_sites*ist.n_down);

   /*Propagate By Multiplying Matrices - Assuming WF Becomes Complex Here*/
   cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,ist.n_sites,ist.n_up,ist.n_sites,1.0,one_body_propagator,ist.n_sites,wf_up,ist.n_up,0.0,temp_wf_up,ist.n_up);
   cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,ist.n_sites,ist.n_down,ist.n_sites,1.0,one_body_propagator,ist.n_sites,wf_down,ist.n_down,0.0,temp_wf_down,ist.n_down);

   /*Copy Wave Function Over*/
   dcopy_2(wf_up,temp_wf_up,ist.n_sites*ist.n_up); 
   dcopy_2(wf_down,temp_wf_down,ist.n_sites*ist.n_down); 

free(temp_wf_up); free(temp_wf_down); 
return; 
}

/*************************************************************************************************/

void neighbors_periodic_boundary_conditions(int *neighbors, int_st ist) {

   /*Determines Nearest Neighbors of a Given Site*/ 

   int site; 
  
   /*Obtain Neighbors for All Sites*/
   for (site = 0; site<ist.n_sites; site++) {

     /*If Site Is Not At Boundary*/
     if (site!=ist.n_sites-1 && site!=0) {
       neighbors[site*2]=site-1;
       neighbors[site*2+1]=site+1;
     } /*If Site Is At Right Boundary*/
     else if (site==ist.n_sites-1){
       neighbors[site*2]=site-1;
       neighbors[site*2+1]=0;
     } /*If Site Is At Left Boundary*/
     else if (site==0) {
       neighbors[site*2]=ist.n_sites-1;
       neighbors[site*2+1]=site+1;
     }
   }

return; 
}

/*************************************************************************************************/
