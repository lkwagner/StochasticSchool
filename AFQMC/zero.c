/*This Contains Functions That Zero Different Vectors and Matrices*/

#include "telluride_basic_cpmc.h"

/*************************************************/

void dzero(double *vec, int size) {

  int i; 

  for ( i=0; i<size; i++) {
     vec[i] = 0.0; 
  } 

return; 
}

/*************************************************/
