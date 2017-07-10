/*This Function Contains Functions That Determine the Energy*/

#include "telluride_basic_cpmc.h"

/******************************************************************************/
double measure_onebody_energy(double *wf_up, double *wf_down, double *trial_wf_up, double *trial_wf_down,int *neighbors, int_st ist, cns_st cns) {

    /*This Function Computes the Hubbard Energy*/
    int sites, j; 
    double kinetic_energy = 0; 
    double *one_body_density_matrix_up, *one_body_density_matrix_down;

    one_body_density_matrix_up = (double *)malloc(ist.n_sites_sq*sizeof(double)); 
    one_body_density_matrix_down = (double *)malloc(ist.n_sites_sq*sizeof(double)); 

    /*Compute the Necessary Density Matrices*/
    compute_density_matrix(wf_up,wf_down,trial_wf_up,trial_wf_down,one_body_density_matrix_up,one_body_density_matrix_down,ist); 

    /*Obtain Kinetic and Potential Energies*/
    for ( sites=0; sites<ist.n_sites; sites++) {

          if ( ist.n_sites > 2 ) {
            for (j=0; j<2; j++) {
               kinetic_energy += one_body_density_matrix_up[sites*ist.n_sites+neighbors[2*sites+j]] + one_body_density_matrix_down[sites*ist.n_sites+neighbors[2*sites+j]]; 
            }
          }
          else {
             for (j=0; j<2; j++) {
               kinetic_energy += .5 * (one_body_density_matrix_up[sites*ist.n_sites+neighbors[2*sites+j]] + one_body_density_matrix_down[sites*ist.n_sites+neighbors[2*sites+j]]);
            }
          }
      }

     
     /*Total Energy - Put Constants In Front*/
     kinetic_energy *= -cns.t * kinetic_energy; 

free(one_body_density_matrix_up); 
free(one_body_density_matrix_down); 
return(kinetic_energy); 
}

/****************************************************************************************************************************/

double measure_total_energy(double *wf_up, double *wf_down, double *trial_wf_up, double *trial_wf_down,int *neighbors, int_st ist, cns_st cns) {

    /*This Function Computes the Hubbard Energy*/
    int sites, j;
    double total_energy, kinetic_energy = 0, potential_energy = 0;
    double *one_body_density_matrix_up, *one_body_density_matrix_down;

    one_body_density_matrix_up = (double *)malloc(ist.n_sites_sq*sizeof(double));
    one_body_density_matrix_down = (double *)malloc(ist.n_sites_sq*sizeof(double));

    /*Compute the Necessary Density Matrices*/
    compute_density_matrix(wf_up,wf_down,trial_wf_up,trial_wf_down,one_body_density_matrix_up,one_body_density_matrix_down,ist);

    /*Obtain Kinetic and Potential Energies*/
    for ( sites=0; sites<ist.n_sites; sites++) {

          potential_energy += one_body_density_matrix_up[sites*ist.n_sites+sites] * one_body_density_matrix_down[sites*ist.n_sites+sites];
          if ( ist.n_sites > 2 ) {
            for (j=0; j<2; j++) {
               kinetic_energy += one_body_density_matrix_up[sites*ist.n_sites+neighbors[2*sites+j]] + one_body_density_matrix_down[sites*ist.n_sites+neighbors[2*sites+j]];
            }
          }
          else {
             for (j=0; j<2; j++) {
               kinetic_energy += .5 * (one_body_density_matrix_up[sites*ist.n_sites+neighbors[2*sites+j]] + one_body_density_matrix_down[sites*ist.n_sites+neighbors[2*sites+j]]);
            }
          }
      }


     /*Total Energy - Put Constants In Front*/
     total_energy = -cns.t * kinetic_energy + cns.U * potential_energy;


free(one_body_density_matrix_up);
free(one_body_density_matrix_down);
return(total_energy);
}

/***********************************************************************************************************/

void compute_density_matrix(double *wf_up,double *wf_down,double *trial_wf_up,double *trial_wf_down,double *density_matrix_up,double *density_matrix_down,int_st ist) {

   /*Computes the Spin Up and Down Greens Functions*/
   double *overlap_up, *overlap_down;
   double *overlap_inverse_up, *overlap_inverse_down;
   double *stored_product_up, *stored_product_down;  

   /*Store Matrices for Product to Be Inverted*/
   overlap_up = (double *)malloc(ist.n_up*ist.n_up*sizeof(double)); 
   overlap_down = (double *)malloc(ist.n_down*ist.n_down*sizeof(double)); 

   overlap_inverse_up = (double *)malloc(ist.n_up*ist.n_up*sizeof(double)); 
   overlap_inverse_down = (double *)malloc(ist.n_down*ist.n_down*sizeof(double)); 

   stored_product_up=(double *)calloc(ist.n_sites*ist.n_up,sizeof(double));
   stored_product_down=(double *)calloc(ist.n_sites*ist.n_down,sizeof(double));

   /*First Construct the Overlap Matrix - Product of Psi_Trial_Transpose and Psi*/
   cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,ist.n_up,ist.n_up,ist.n_sites,1.0,trial_wf_up,ist.n_up,wf_up,ist.n_up,0.0,overlap_up,ist.n_up); 
   cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,ist.n_down,ist.n_down,ist.n_sites,1.0,trial_wf_down,ist.n_down,wf_down,ist.n_down,0.0,overlap_down,ist.n_down);

   /*Need to Get Inverse Matrix!*/
   inverse(overlap_up,overlap_inverse_up,ist.n_up); 
   inverse(overlap_down,overlap_inverse_down,ist.n_down);  

   /*Get Up Green's Function First*/
   cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,ist.n_up,ist.n_sites,ist.n_up,1.0,overlap_inverse_up,ist.n_up,trial_wf_up,ist.n_up,0.0,stored_product_up,ist.n_sites);
   cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,ist.n_sites,ist.n_sites,ist.n_up,1.0,wf_up,ist.n_up,stored_product_up,ist.n_sites,0.0,density_matrix_up,ist.n_sites);

   /*Get Down Green's Function*/
   cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,ist.n_down,ist.n_sites,ist.n_down,1.0,overlap_inverse_down,ist.n_down,trial_wf_down,ist.n_down,0.0,stored_product_down,ist.n_sites);
   cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,ist.n_sites,ist.n_sites,ist.n_down,1.0,wf_down,ist.n_down,stored_product_down,ist.n_sites,0.0,density_matrix_down,ist.n_sites);

free(overlap_up); 
free(overlap_down); 
free(overlap_inverse_up); 
free(overlap_inverse_down); 
free(stored_product_up);
free(stored_product_down);
return;
}

/************************************************************************************************************************************/

void inverse(double *matrix, double *inverse, int size) {

     int i,j;
     int *indx;
     double *col;
     double d;

     indx=(int *)calloc(size,sizeof(int));
     col=(double *)calloc(size,sizeof(double));

     ludmp(matrix,size,indx,&d);

     for (j=0; j<size; j++) {
        for (i=0; i<size; i++) {
           col[i]=0.0;
        }
        col[j]=1.0;

        lubksb(matrix,size,indx,col);
        for(i=0; i<size; i++) {
          inverse[i*size+j]=col[i];
        }
    }

free(col);
free(indx);
return;
}


/************************************************************************************************************************************/

void ludmp(double *matrix, int n, int *indx, double *d) {

   /*Performs an LU Decomposition Based on Numerical Recipes*/

   int i,imax=1,j,k;
   double big,dum,sum,temp;
   double *vv;

   vv=(double*)calloc(n,sizeof(double));

   *d=1.0;
   for (i=1; i<=n; i++){
     big=0.0;
     for (j=1; j<=n; j++) {
        if ((temp=fabs(matrix[(i-1)*n+j-1])) > big) big=temp;
     }
     vv[i-1]=1.0/big;
    }

    for (j=1; j<=n; j++) {

      for (i=1; i<j; i++) {
         sum=matrix[(i-1)*n+j-1];
         for (k=1; k<i; k++) {
           sum -= matrix[(i-1)*n+k-1] * matrix[(k-1)*n+j-1];
         }
         matrix[(i-1)*n+j-1]=sum;
       }

    big=0.0;
    for (i=j; i<=n; i++) {
       sum=matrix[(i-1)*n+j-1];
       for (k=1; k<j; k++) {
         sum -= matrix[(i-1)*n+k-1]*matrix[(k-1)*n+j-1];
       }
       matrix[(i-1)*n+j-1]=sum;

       if ( (dum=vv[i-1]*fabs(sum)) >= big) {
       big = dum;
          imax = i;
       }
     }

     if ( j!= imax ) {
        for (k=1; k<=n; k++) {
           dum = matrix[(imax-1)*n+k-1];
           matrix[(imax-1)*n+k-1] = matrix[(j-1)*n+k-1];
           matrix[(j-1)*n+k-1]=dum;
         }
         (*d) = -(*d);
         vv[imax-1]=vv[j-1];
     }
     indx[j-1]=imax;

     if (matrix[(j-1)*n+j-1] == 0.0 ) {
         matrix[(j-1)*n+j-1] = TINY;
     }

     if (j != n ) {
        dum=1.0/matrix[(j-1)*n+j-1];
        for (i=j+1; i<=n; i++) {
          matrix[(i-1)*n+j-1] *= dum;
        }
      }

   }

free(vv);
return;
}

/********************************************************************************************************************************/

void lubksb(double *matrix, int n, int *indx, double *b) {

    /*Peforms LU Forward and Backward Substitution From NRC*/

    int i,ii=0,ip,j;
    double sum;

    for (i=1; i<=n; i++) {
       ip=indx[i-1];
       sum=b[ip-1];
       b[ip-1]=b[i-1];

        if (ii)
              for (j=ii; j<=i-1; j++) sum -= matrix[(i-1)*n+j-1] * b[j-1];
        else if (sum) ii=i;
        b[i-1]=sum;
     }

     for (i=n; i>=1; i--) {
        sum=b[i-1];
        for (j=i+1; j<=n; j++) sum -= matrix[(i-1)*n+j-1]*b[j-1];
        b[i-1]=sum/matrix[(i-1)*n+i-1];
     }

return;
}




