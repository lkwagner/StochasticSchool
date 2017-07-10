/*This File Contains Routines to Orthogonalize the Columns*/

#include "telluride_basic_cpmc.h"

/****************************************************************************************************/

void orthogonalize(double *wf_up,double *wf_down,int_st ist){

    /*Orthogonalizes Wave Function After Using Random Numbers to Determine To Prevent Overlaps*/

    double *R_up, *R_down;

    R_up = (double *)malloc(ist.n_sites_sq*sizeof(double));
    R_down = (double *)malloc(ist.n_sites_sq*sizeof(double));

    /*Performs Modified Gram Schmidt to Orthogonalize*/
    modified_gram_schmidt(wf_up,R_up,ist.n_sites,ist.n_up);
    modified_gram_schmidt(wf_down,R_down,ist.n_sites,ist.n_down);

free(R_up);
free(R_down);
return;
}

/****************************************************************************************************/

void modified_gram_schmidt(double *q, double *r, int size1, int size2) {

   /*The Modified Gram-Schmidt Routine Used to Stabilize Matrix Products*/
   int i, j, k;
   double temporary, anorm;
   double *d;

   d=(double *)calloc(size2,sizeof(double));
   dzero(r,size2*size2);

   for (i=1; i<=size2; i++) {

     temporary = 0.0;
     for (j=1; j<=size1; j++) {
       temporary += q[(j-1)*size2+i-1]*q[(j-1)*size2+i-1];
     }
     d[i-1]=sqrt(temporary);
     anorm = 1.0/d[i-1];

     for (j=1; j<=size1; j++) {
       q[(j-1)*size2+i-1] *= anorm;
     }

     for ( j=i+1; j<=size2; j++) {
       temporary = 0.0;
       for (k=1; k<=size1; k++) {
         temporary += q[(k-1)*size2+i-1] * q[(k-1)*size2+j-1];
       }

       for (k=1; k<=size1; k++) {
         q[(k-1)*size2+j-1]-= temporary * q[(k-1)*size2+i-1];
       }
       r[(i-1)*size2+j-1]=temporary*anorm;
      }

   } /*i loop*/

   /*Now Make V Into R Matrix*/
   for (i=0; i<size2; i++) {
    r[i*size2+i]=d[i];
   }

   free(d);

return;
}

/*************************************************************************************/
