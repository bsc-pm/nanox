/*************************************************************************************/
/*      Copyright 2010 Barcelona Supercomputing Center                               */
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

#ifndef _OMP_H_DEF
#define _OMP_H_DEF

/* OpenMP API interface */

/*
* define the lock data types
*/
typedef void * omp_lock_t;
typedef void * omp_nest_lock_t;

/*
* define the schedule kinds
*/
typedef enum omp_sched_t {
   omp_sched_static = 1,
   omp_sched_dynamic = 2,
   omp_sched_guided = 3,
   omp_sched_auto = 4
} omp_sched_t;

/*
* exported OpenMP functions
*/
#ifdef __cplusplus
extern "C"
{
#endif

int omp_get_num_threads ( void );
int omp_get_thread_num ( void );
int omp_get_num_procs ( void );
int omp_in_parallel ( void );
int omp_get_thread_limit ( void );
void omp_set_max_active_levels ( int max_active_levels );
int omp_get_max_active_levels ( void );

#ifdef __cplusplus
}

#endif

#endif

