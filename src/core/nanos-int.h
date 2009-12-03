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

#ifndef __NANOS_INT_H
#define __NANOS_INT_H

#include <stdio.h>

// C++ types hidden as void *
typedef void * nanos_thread_t;

typedef struct {
   int lower;
   int upper;
   int step;
} nanos_loop_info_t;

typedef struct {
   bool mandatory_creation:1;
   bool tied:1;
   bool reserved0:1;
   bool reserved1:1;
   bool reserved2:1;
   bool reserved3:1;
   bool reserved4:1;
   bool reserved5:1;
   nanos_thread_t * tie_to;
   unsigned int priority;
} nanos_wd_props_t;

typedef struct {
  void * (*factory) (void *prealloc, void *arg);
  size_t dd_size;
  void * arg;
} nanos_device_t;

#endif
