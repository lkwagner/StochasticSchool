/*This Code Relates to Acquiring Parameters from Input Files and Storing Them*/

#include "telluride_basic_cpmc.h"

/*Get Parameters*****************************************************************************************/
void get_parameters(int_st *ist,cns_st *cns) {

  FILE *pf = fopen("telluride_basic_cpmc.par", "r");
  FILE *po = fopen("telluride_basic_cpmc.dat", "w+");

  /*Scan In Relevant Parameters*/

  /*Relevant Parameters for Repulsive Hubbard Model*/
  fscanf (pf,"%d",&ist->n_sites);   /*Number of Sites in Dimension 1*/

  /*Get Sites Squared*/
  ist->n_sites_sq = ist->n_sites * ist->n_sites; 

  /*Number of Electrons Up and Down*/
  fscanf (pf,"%d",&ist->n_up);    /*Number of Electrons Up*/
  fscanf (pf,"%d",&ist->n_down);  /*Number of Electrons Down*/

  /*Number of Steps*/
  fscanf(pf,"%d",&ist->n_steps); /*Number of Iterative Steps*/  

  /*Hubbard Simulation Parameters*/
  fscanf (pf,"%lf",&cns->U);
  fscanf (pf,"%lf",&cns->t);
  fscanf (pf,"%lf",&cns->delta_tau); 

  /*Print Out Parameters to File*/
  fprintf(po,"Number of Sites: %d\n", ist->n_sites); 
  fprintf(po,"Number of Up Electrons: %d\n", ist->n_up); 
  fprintf(po,"Number of Down Electrons: %d\n", ist->n_down); 
 
  fprintf(po,"Number of CPMC Steps: %d\n", ist->n_steps); 

  fprintf(po,"U: %f\n", cns->U); 
  fprintf(po,"t: %f\n", cns->t); 
  fprintf(po,"Delta tau: %f\n", cns->delta_tau); 
  fflush(po);    

  /*Set Value of Exponential Constant for HS Transformation*/
  cns->exponential_constant = sqrt(cns->U * cns->delta_tau);  

fclose(po); 
fclose(pf); 
return; 
}
