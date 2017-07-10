/*This is a very basic CPMC Code for the 4-Site Linear Hubbard Model That Incorporates Building the Kinetic Matrix and Potential Operators Starting with a Non-Interacting Trial Wave Function - Assumes Linear and Hubbard, But Takes in U, t, and the Number of Sites; Outputs Energy Measurements - July 2017*/

/*Need to Load BLAS and LAPACK to Run*/

#include "telluride_basic_cpmc.h"

int main() { 

   /*Output File*/
   FILE *pout = fopen("energy.dat", "a+"); 

   /*Set Up Structures to Hold Constants for Simulation*/
   cns_st cns; int_st ist;
  
   /*Obtain Simulation Control Parameters*/
   get_parameters(&ist,&cns);

   /*Necessary Matrices*/
   int *neighbors; 
   int steps; 
   double *one_body_matrix, *one_body_propagator;
   double *trial_wf_up, *trial_wf_down; 
   double *wf_up, *wf_down;   
   double total_energy = 0; 
 
   neighbors = (int *)malloc(ist.n_sites*2*sizeof(int)); 
   one_body_matrix = (double *)malloc(ist.n_sites_sq*sizeof(double)); 
   one_body_propagator = (double *)malloc(ist.n_sites_sq*sizeof(double)); 
   trial_wf_up = (double *)malloc(ist.n_sites*ist.n_up*sizeof(double)); 
   trial_wf_down = (double *)malloc(ist.n_sites*ist.n_down*sizeof(double));  
   wf_up = (double *)malloc(ist.n_sites*ist.n_up*sizeof(double)); 
   wf_down = (double *)malloc(ist.n_sites*ist.n_down*sizeof(double)); 

   /*Form One-Body Matrix - Just Kinetic Piece for Now, Assuming PBCs*/
   form_one_body(one_body_matrix,neighbors,ist,cns); 
   
   /*Exponentiate Matrix and Obtain Non-Interacting Trial WFs*/
   exponentiate_one_body(one_body_propagator,one_body_matrix,trial_wf_up,trial_wf_down,ist,cns);   
 
   /*Copy Trial Wave Function to Wave Function To Start Iteration*/
   dcopy_2(wf_up,trial_wf_up,ist.n_sites*ist.n_up); 
   dcopy_2(wf_down,trial_wf_down,ist.n_sites*ist.n_down); 

   /*Now Run Through Steps Until Energy Equilibrates and Fluctuates*/
   for (steps = 0; steps < ist.n_steps; steps++) {

     /*Propagate One-Body Part By a Half-Step*/
     propagate_one_body(one_body_propagator,wf_up,wf_down,ist); 

     /*Propagate Using HS Transformation*/
     propagate_two_body(wf_up,wf_down,ist,cns); 

     /*Propagate One-Body Part By a Half-Step*/     
     propagate_one_body(one_body_propagator,wf_up,wf_down,ist);

     /*Orthogonalize Every So Often As Well, So That Columns of WF Do Not Converge to the Same Value*/
     if ( steps%10 == 0 ) {
       orthogonalize(wf_up,wf_down,ist); 
     } 

     /*Every 20 Steps, Print the Energy*/
     if ( steps%20 == 0 ) {
        total_energy = measure_total_energy(wf_up,wf_down,trial_wf_up,trial_wf_down,neighbors,ist,cns);
        fprintf(pout, "%d\t %f\n", steps, total_energy); fflush(pout); 
     } 

     int i, j; 
     printf("steps %d\n", steps); 
     for (i=0; i<ist.n_sites; i++) {
      for (j=0; j<ist.n_up; j++) {
       printf("%f\t", wf_up[i*ist.n_up+j]); fflush(NULL);  
      } printf("\n"); 
     } printf("\n"); fflush(NULL); 

     for (i=0; i<ist.n_sites; i++) {
      for (j=0; j<ist.n_down; j++) {
       printf("%f\t", wf_down[i*ist.n_down+j]); fflush(NULL);
      } printf("\n");
     } printf("\n"); fflush(NULL); 


   } 

   /*Free Matrices*/
   free(one_body_matrix); free(one_body_propagator); 
   free(trial_wf_up); free(trial_wf_down);
   free(wf_up); free(wf_down);  
   free(neighbors); 
   fclose(pout); 

return 0; 
}
