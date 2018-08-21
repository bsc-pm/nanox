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

#pragma omp task inout([ts][ts]A)
void omp_potrf(double * const A, int ts, int ld)
{
	static int INFO;
	static char L = 'L';
	dpotrf_(&L, &ts, A, &ld, &INFO);
}

#pragma omp task in([ts][ts]A) inout([ts][ts]B)
void omp_trsm(double *A, double *B, int ts, int ld)
{
	static char LO = 'L', TR = 'T', NU = 'N', RI = 'R';
	static double DONE = 1.0;
	dtrsm_(&RI, &LO, &TR, &NU, &ts, &ts, &DONE, A, &ld, B, &ld );
}

#pragma omp task in([ts][ts]A) inout([ts][ts]B)
void omp_syrk(double *A, double *B, int ts, int ld)
{
	static char LO = 'L', NT = 'N';
	static double DONE = 1.0, DMONE = -1.0;
	dsyrk_(&LO, &NT, &ts, &ts, &DMONE, A, &ld, &DONE, B, &ld );
}

void gemm_tile(double *A, double *B, double *C, int ts, int ld)
{
	static char TR = 'T', NT = 'N';
	static double DONE = 1.0, DMONE = -1.0;
	dgemm_(&NT, &TR, &ts, &ts, &ts, &DMONE, A, &ld, B, &ld, &DONE, C, &ld);
}

#pragma omp task in([super][super]A, [super][super]B) inout([super][super]C)
void omp_gemm(double *A, double *B , double *C ,int super, int region)
{
    int i, j, k;
    
    for(k=0; k<super ;k+=region)
    {
        for(i=0; i<super;i+=region)
        {
            for(j=0; j<super;j+=region)
            {
             gemm_tile(&A[k*super+i], &B[k*super+j], &C[j*super+i], super, region);    
            }
        }
    }
}

void cholesky_blocked(const int ts, const int nt, double* Ah[nt][nt])
{
	for (int k = 0; k < nt; k++) {

		// Diagonal Block factorization
		omp_potrf (Ah[k][k], ts, ts);

		// Triangular systems
		for (int i = k + 1; i < nt; i++) {
			omp_trsm (Ah[k][k], Ah[k][i], ts, ts);
		}

		// update trailing matrix
		for (int i = k + 1; i < nt; i++) {
			for (int j = k + 1; j < i; j++) {
				omp_gemm (Ah[k][i], Ah[k][j], Ah[j][i], ts, ts);
			}
			omp_syrk (Ah[k][i], Ah[i][i], ts, ts);
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
	double * const matrix = (double *) malloc(n * n * sizeof(double));
	assert(matrix != NULL);

	// Init matrix
	initialize_matrix(n, ts, matrix);

	// Allocate matrix
	double * const original_matrix = (double *) malloc(n * n * sizeof(double));
	assert(original_matrix != NULL);

	const int nt = n / ts;

	// Allocate blocked matrix
	double *Ah[nt][nt];

	for (int i = 0; i < nt; i++) {
		for (int j = 0; j < nt; j++) {
			Ah[i][j] = malloc(ts * ts * sizeof(double));
			assert(Ah[i][j] != NULL);
		}
	}

	for (int i = 0; i < n * n; i++ ) {
		original_matrix[i] = matrix[i];
	}

	printf ("Executing ...\n");
	convert_to_blocks(ts, nt, n, (double(*)[n]) matrix, Ah);

	const float t1 = get_time();
	cholesky_blocked(ts, nt, (double* (*)[nt]) Ah);

	const float t2 = get_time() - t1;
	convert_to_linear(ts, nt, n, Ah, (double (*)[n]) matrix);

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

	// Free blocked matrix
	for (int i = 0; i < nt; i++) {
		for (int j = 0; j < nt; j++) {
			assert(Ah[i][j] != NULL);
			free(Ah[i][j]);
		}
	}

	free(matrix);
	return 0;
}

