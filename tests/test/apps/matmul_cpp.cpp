/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

/*
<testinfo>
test_generator=gens/mixed-generator
</testinfo>
*/

#include <math.h>
#include "config.hpp"
#include "nanos.h"
#include <iostream>
#include "smpprocessor.hpp"
#include "system.hpp"

using namespace std;

using namespace nanos;
using namespace nanos::ext;

#define USE_NANOS     true
#define VECTOR_SIZE    16 
#define BSIZE          64

typedef struct {
   nanos_loop_info_t loop_info;
} main__loop_1_data_t;

struct matrix_block
{
   int i;
   int j;
   int k;
};


typedef double block_t[BSIZE][BSIZE];

void main__loop_1 ( void *args )
{
   int i, j, k;
   struct matrix_block *myBlock = (struct matrix_block *) args;
   //double (* (*_A)[BSIZE])[BSIZE];
   //double (* (*_B)[BSIZE])[BSIZE];
   //double (* (*_C)[BSIZE])[BSIZE];

   block_t *_A;
   block_t *_B;
   block_t *_C;

   WD *wd = myThread->getCurrentWD();

   CopyData* cd = wd->getCopies();
   //_A = (double (* (*)[BSIZE])[BSIZE]) cd[0].getAddress();
   //_B = (double (* (*)[BSIZE])[BSIZE]) cd[1].getAddress();
   //_C = (double (* (*)[BSIZE])[BSIZE]) cd[2].getAddress();
   _A = (block_t *) cd[0].getAddress();
   _B = (block_t *) cd[1].getAddress();
   _C = (block_t *) cd[2].getAddress();

   //fprintf(stderr, "[node %d] _A = %p\n", sys.getNetwork()->getNodeNum(), _A);
   //fprintf(stderr, "[node %d] _B = %p\n", sys.getNetwork()->getNodeNum(), _B);
   //fprintf(stderr, "[node %d] _C = %p\n", sys.getNetwork()->getNodeNum(), _C);

   for (i = 0; i < BSIZE; i++)
      for (j = 0; j < BSIZE; j++)
         for (k = 0; k < BSIZE; k++)
            (*_C)[i][j] += (*_A)[i][k] * (*_B)[k][j];

   //fprintf(stderr, "[node %d] executing block %d %d %d\n", sys.getNetwork()->getNodeNum(), myBlock->i, myBlock->j, myBlock->k);
   //{ int _x = 0; while (_x < 1000000) _x++; }
}

void init_matrix(double *A[VECTOR_SIZE][VECTOR_SIZE],double *B[VECTOR_SIZE][VECTOR_SIZE], double *C[VECTOR_SIZE][VECTOR_SIZE])
{
  int i, j, ii, jj;
  
  /* vandermonde + uppertriangular */
  for (i = 0; i < VECTOR_SIZE; i++) {
    for (ii = 0; ii < BSIZE; ii++)
      for (j = 0; j < VECTOR_SIZE; j++)
	for (jj = 0; jj < BSIZE; jj++){
	  A[i][j][ii*BSIZE+jj] = pow (1.0/(double)((i*BSIZE)+ii+1), (double)(j*BSIZE)+jj);
	  B[i][j][ii*BSIZE+jj] = ((i*BSIZE)+ii)<=((j*BSIZE)+jj);
	  C[i][j][ii*BSIZE+jj] = 0;
	}
  }
  return;
}

void verify(int niter, double *C[VECTOR_SIZE][VECTOR_SIZE])
{
   int i,j,ii,jj;
   for (i=0; i<VECTOR_SIZE; i++)
      for (j=0; j<VECTOR_SIZE; j++)
         for (ii=0; ii<BSIZE; ii++)
            for (jj=0; jj<BSIZE; jj++)
            {
               if((i*BSIZE+ii)<2) continue;
               double i1 = 1.0/((double)(i*BSIZE)+ii+1);
               double shb = niter * (pow (i1, (double)((j*BSIZE)+jj+1))-1) / (i1-1);
               double diff = (C[i][j][ii*BSIZE+jj]-shb)/shb; if (diff < 0) diff = -diff;

               if (diff > 1e-4)
               {
                  fprintf(stderr,"Verification FAILED %f %f \n", C[i][j][ii*BSIZE+jj],shb);
                  return;
               }
            }
}

void write_matrix (double *C[VECTOR_SIZE][VECTOR_SIZE]) {
  int i, j, ii, jj;
  //printf("%s\n",__FUNCTION__);

  fprintf (stderr, "%d\n %d\n", VECTOR_SIZE * BSIZE, VECTOR_SIZE * BSIZE);
  for (i = 0; i < VECTOR_SIZE; i++)
    for (ii = 0; ii < BSIZE; ii++)
      {
	for (j = 0; j < VECTOR_SIZE; j++)
	  for (jj = 0; jj < BSIZE; jj++)
	    fprintf (stderr, "%f ", C[i][j][ii*BSIZE+jj]);
	fprintf (stderr, "\n");
      }

}

int main ( int argc, char **argv )
{
   int i, j, k;
   double **A, **B, **C;
   bool check = true;

   main__loop_1_data_t _loop_data;
   main__loop_1_data_t _loop_data2;

   WG *wg = myThread->getCurrentWD();

   A = new double *[VECTOR_SIZE * VECTOR_SIZE];
   B = new double *[VECTOR_SIZE * VECTOR_SIZE];
   C = new double *[VECTOR_SIZE * VECTOR_SIZE];
   char *copies = new char[ sizeof( CopyData ) * 3 * VECTOR_SIZE * VECTOR_SIZE * VECTOR_SIZE ];
   struct matrix_block *blocks = new struct matrix_block[ VECTOR_SIZE * VECTOR_SIZE * VECTOR_SIZE ];

  for (i = 0; i < VECTOR_SIZE*VECTOR_SIZE; i++)
  {
     A[i] = new double[BSIZE*BSIZE];
     B[i] = new double[BSIZE*BSIZE];
     C[i] = new double[BSIZE*BSIZE];

     //fprintf(stderr, "A[%d] = %p\n", i, A[i]);
     //fprintf(stderr, "B[%d] = %p\n", i, B[i]);
     //fprintf(stderr, "C[%d] = %p\n", i, C[i]);
  }

  init_matrix( ( double *(*)[VECTOR_SIZE] ) A, ( double *(*)[VECTOR_SIZE] ) B, ( double *(*)[VECTOR_SIZE] ) C );
  int wdc = 0;

        for (k = 0; k < VECTOR_SIZE; k++)
        {
  for (i = 0; i < VECTOR_SIZE; i++)
  {
     for (j = 0; j < VECTOR_SIZE; j++)
     {
           blocks[wdc].i = i;
           blocks[wdc].j = j;
           blocks[wdc].k = k;
           //char *copies = new char[ sizeof( CopyData ) * 3 ];
           CopyData *current_cd3 = reinterpret_cast<CopyData *>( &copies[ sizeof( CopyData ) * 3 * ( i * VECTOR_SIZE * VECTOR_SIZE + j * VECTOR_SIZE + k ) ] );
           new ( &current_cd3[0] ) CopyData( (uint64_t) A[i * VECTOR_SIZE + k], NANOS_SHARED, true, false, sizeof( double ) * BSIZE * BSIZE );
           new ( &current_cd3[1] ) CopyData( (uint64_t) B[k * VECTOR_SIZE + j], NANOS_SHARED, true, false, sizeof( double ) * BSIZE * BSIZE );
           new ( &current_cd3[2] ) CopyData( (uint64_t) C[i * VECTOR_SIZE + j], NANOS_SHARED, true, true,  sizeof( double ) * BSIZE * BSIZE );
           WD * wd = new WD( new SMPDD( main__loop_1 ), sizeof( struct matrix_block ), &blocks[wdc], 3, (CopyData *) &( copies[ sizeof( CopyData ) * 3 * ( i * VECTOR_SIZE * VECTOR_SIZE + j * VECTOR_SIZE + k )] ) );
           wdc++;
           wg->addWork( *wd );
           sys.submit( *wd );
        }
     //wg->waitCompletion();
     }
     wg->waitCompletion();
  }
   fprintf(stderr, "LOOP DONE wd count %d!\n", wdc);
   wg->waitCompletion();

   //write_matrix( (double* (*)[VECTOR_SIZE])C);
   verify(1, (double* (*)[VECTOR_SIZE]) C );
   fprintf(stderr,"Verification ok\n");
}

