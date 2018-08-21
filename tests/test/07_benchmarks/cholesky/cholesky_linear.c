/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include "util.h"

/*
<testinfo>
test_generator=gens/mcc-openmp-generator
test_generator_ENV=( "NX_TEST_MODE=performance" )
test_LDFLAGS="-lmkl_sequential -lmkl_core"
test_ignore_fail=yes
</testinfo>
*/

#pragma omp task inout((*A)[i;ts][j;ts])
void omp_potrf(int ts, int n, double (* const A)[n][n], int i, int j)
{
	static int INFO;
	static char L = 'L';
	dpotrf_(&L, &ts, &(*A)[i][j], &n, &INFO);
}

#pragma omp task in((*A)[ai;ts][aj;ts]) inout((*B)[bi;ts][bj;ts])
void omp_trsm(int ts, int n, double (*A)[n][n], int ai, int aj, double (*B)[n][n], int bi, int bj)
{
	static char LO = 'L', TR = 'T', NU = 'N', RI = 'R';
	static double DONE = 1.0;
	dtrsm_(&RI, &LO, &TR, &NU, &ts, &ts, &DONE, &(*A)[ai][aj], &n, &(*B)[bi][bj], &n );
}

#pragma omp task in((*A)[ai;ts][aj;ts]) inout((*B)[bi;ts][bj;ts])
void omp_syrk(int ts, int n, double (*A)[n][n], int ai, int aj, double (*B)[n][n], int bi, int bj)
{
	static char LO = 'L', NT = 'N';
	static double DONE = 1.0, DMONE = -1.0;
	dsyrk_(&LO, &NT, &ts, &ts, &DMONE, &(*A)[ai][aj], &n, &DONE, &(*B)[bi][bj], &n );
}

#pragma omp task in((*A)[ai;ts][aj;ts], (*B)[bi;ts][bj;ts]) inout((*C)[ci;ts][cj;ts])
void omp_gemm(int ts, int n, double (*A)[n][n], int ai, int aj, double (*B)[n][n], int bi, int bj, double (*C)[n][n], int ci, int cj)
{
	static char TR = 'T', NT = 'N';
	static double DONE = 1.0, DMONE = -1.0;
	dgemm_(&NT, &TR, &ts, &ts, &ts, &DMONE, &(*A)[ai][aj], &n, &(*B)[bi][bj], &n, &DONE, &(*C)[ci][cj], &n);
}

void cholesky_linear(const int ts, const int n, double (*A)[n][n])
{
	for (int k = 0; k < n; k += ts) {

		// Diagonal Block factorization
		omp_potrf(ts, n, A, k, k);

		// Triangular systems
		for (int i = k + ts; i < n; i += ts) {
			omp_trsm (ts, n, A, k, k, A, k, i);
		}

		// update trailing matrix
		for (int i = k + ts; i < n; i += ts) {
			for (int j = k + ts; j < i; j += ts) {
				omp_gemm (ts, n, A, k, i, A, k, j, A, j, i);
			}
			omp_syrk (ts, n, A, k, i, A, i, i);
		}
	}
#pragma omp taskwait
}

//--------------------------- MAIN --------------------
int main(int argc, char* argv[])
{

	const double eps = BLAS_dfpinfo( blas_eps );

	if ( argc != 3) {
		printf( "cholesky size block_size\n" );
		exit( -1 );
	}
	const int n = atoi(argv[1]); // n matrix size
	const int ts = atoi(argv[2]);  // super tile size

	// Allocate matrix
	double (* const matrix)[n][n] = (double (*)[n][n]) malloc(n * n * sizeof(double));
	assert(matrix != NULL);

	// Allocate matrix
	double (* const original_matrix)[n][n] = (double (*)[n][n]) malloc(n * n * sizeof(double));
	assert(original_matrix != NULL);

	// Init matrix
	initialize_matrix(n, ts, original_matrix);


   for (int i = 0; i < n; i += ts ) {
#pragma omp task in( (*original_matrix)[i;ts][0;n] ) out( (*matrix)[i;ts][0;n] ) firstprivate(n, i, ts)
      for (int ii = i; ii < i+ts; ii++ ) {
         for (int j = 0; j < n; j += 1 ) {
            (*matrix)[ii][j] = (*original_matrix)[ii][j];
         }
      }
   }

   

#pragma omp taskwait noflush

	const float t1 = get_time();
	cholesky_linear(ts, n, matrix);

	const float t2 = get_time() - t1;

	const char uplo = 'L';
	const int info_factorization = check_factorization( n, original_matrix, matrix, n, uplo, eps);

	free(original_matrix);

	float time = t2;
	float gflops = (((1.0 / 3.0) * n * n * n) / ((time) * 1.0e+9));

	// Print results
	printf( "============ CHOLESKY RESULTS ============\n" );
	printf( "  matrix size:          %dx%d\n", n, n);
	printf( "  block size:           %dx%d\n", ts, ts);
	printf( "  time (s):             %f\n", time);
	printf( "  performance (gflops): %f\n", gflops);
	printf( "==========================================\n" );

	free(matrix);
	return 0;
}

