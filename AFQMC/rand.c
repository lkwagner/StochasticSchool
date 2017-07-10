/*This Contains the Necessary Random Number Generators*/

#include "telluride_basic_cpmc.h"

/*******************************************************************************/

float gasdev(long *idum) {
   /*Returns a Random Number from a Gaussian Distribution*/

static int iset=0;
static float gset;
float fac, rsq, v1, v2;

if (iset ==0) {
   do {
    v1=2.0*ran1(idum)-1.0;
    v2=2.0*ran1(idum)-1.0;
    rsq=v1*v1+v2*v2;
   }   while (rsq >=1.0 || rsq==0.0);
   fac=sqrt(-2.0*log(rsq)/rsq);
   gset=v1*fac;
   iset=1;
   return v2*fac;
 }
 else {
   iset=0;
   return gset;
 }

}

/**************************************************************************/

void Randomize()
{
  double tt = (double)time(0);
  long   rn = (long)tt % 1000;
  int    i;
  
  for (i = 0; i < rn; i++) random();
  return;
}

/****************************************************************************/

float ran1(long *idum)
{
int j;
long k;
float temp;
static long iy=0;
static long iv[NTAB];

if (*idum <= 0 || !iy) {
   if (-(*idum) < 1) *idum=1;
   else *idum = -(*idum);
  for (j=NTAB+7;j>=0;j--) {
     k=(*idum)/IQ;
      *idum=IA*(*idum-k*IQ)-IR*k;
     if (*idum < 0) *idum += IM;
     if (j < NTAB) iv[j] = *idum;
   }
   iy=iv[0];
  }
   k=(*idum)/IQ;
   *idum=IA*(*idum-k*IQ)-IR*k;
   if (*idum < 0) *idum += IM;
    j=iy/NDIV;
    iy=iv[j];
    iv[j] = *idum;
    if ((temp=AM*iy) > RNMX) return RNMX;
    else return temp;

}

