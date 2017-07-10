/*This File Contains All Functions Related to Propagating By the Two-Body Operator*/

#include "telluride_basic_cpmc.h"

/*************************************************************************************/
void propagate_two_body(double *wf_up,double *wf_down,int_st ist,cns_st cns){

  /*Propagate By the Two-Body Operator By Performing HS Transformation and Applying Operator*/

  int sites; 
  int i; 
  long tidum; 
  double field; 
  double exponential_twobody_up, exponential_twobody_down; 
  double exponential_onebody; 

  /*Quick Random Number Generator*/
  Randomize(); tidum = -random();

  /*Note That In a More Advanced Code, the One-Body Operator Coming from the HS Transformation Would Just Be Combined with the Original One Body Operator at the Beginning; I Am Not Doing That Here*/

  /*Get One Body Exponential Term*/
  exponential_onebody = exp(-cns.delta_tau * cns.U/2.0);

  /*Run Through All Sites Obtaining Fields and Then Applying Them*/
  for (sites = 0; sites<ist.n_sites; sites++) {

    /*Generate Random Field From Gaussian Distribution*/
    field = gasdev(&tidum);  

    /*Obtain Exponentials That Multiply Row i of Wave Function to Act on Density i*/
    exponential_twobody_up = exp(field * cns.exponential_constant);   
    exponential_twobody_down = exp(-1 * field * cns.exponential_constant); 

    /*Multiply Exponential Into Wave Function - Replace with BLAS Routine*/
    for (i=0; i<ist.n_up; i++) {
      wf_up[sites*ist.n_up+i] *= exponential_onebody * exponential_twobody_up; 
    }

    for (i=0; i<ist.n_down; i++) {
      wf_down[sites*ist.n_down+i] *= exponential_onebody * exponential_twobody_down; 
    }  

  }

return; 
}
