/*This Contains Functions That Copies Different Vectors and Matrices*/

#include "telluride_basic_cpmc.h"

/*************************************************/

void dcopy_2(double *new_vec, double *old_vec, int size) {

  /*Copies The Old Vector Into the New Vector*/
  int i; 

  for (i=0; i<size; i++) {
    new_vec[i] = old_vec[i]; 
  }

return; 
}
