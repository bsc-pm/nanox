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
#include <stdint.h>
#include <stddef.h>

typedef struct {
   void **address;
   ptrdiff_t offset;
   struct {
     bool  input: 1;
     bool  output: 1;
     bool  can_rename:1;
   } flags;
   size_t  size;
} nanos_dependence_internal_t;

typedef enum {
   NANOS_PRIVATE,
   NANOS_SHARED,
} nanos_sharing_t;

typedef struct {
   uint64_t address;
   nanos_sharing_t sharing;
   struct {
      bool input: 1;
      bool output: 1;
   } flags;
   size_t size;
} nanos_copy_data_internal_t;

#ifndef _NANOS_INTERNAL

typedef nanos_dependence_internal_t nanos_dependence_t;
typedef nanos_copy_data_internal_t nanos_copy_data_t;

#else

namespace nanos {
   class Dependency;
   class CopyData;
}
typedef nanos::Dependency nanos_dependence_t;
typedef nanos::CopyData nanos_copy_data_t;

#endif

// SlicerDataFor: related structures
typedef struct {
   int _lower;  /**< Loop lower bound */
   int _upper;  /**< Loop upper bound */
   int _step;   /**< Loop step */
   int _chunk;  /**< Slice chunk */
   int _sign;   /**< Loop sign 1 ascendant, -1 descendant */
} nanos_slicer_data_for_internal_t;

#ifndef _NANOS_INTERNAL

typedef nanos_slicer_data_for_internal_t           nanos_slicer_data_for_t;

#else

namespace nanos {
   class SlicerDataFor;
}
typedef nanos::SlicerDataFor          nanos_slicer_data_for_t;

#endif

#if 0
typedef struct {
   int _nWD;    /**< Number of WorkDescriptors */
} nanos_slicer_data_compound_wd_internal_t;
#endif

// C++ types hidden as void *
typedef void * nanos_thread_t;
typedef void * nanos_wd_t;                                                                                                                               

// SlicerDataCompoundWD: related structures
typedef struct {
   int nsect;
   nanos_wd_t lwd[];
} nanos_compound_wd_data_t;

typedef struct {
   int lower;
   int upper;
   int step;
   bool last;
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
   nanos_thread_t tie_to;
   unsigned int priority;
} nanos_wd_props_t;

typedef struct {
  void * (*factory) (void *prealloc, void *arg);
  size_t dd_size;
  void * arg;
} nanos_device_t;

// instrumentor structures

typedef enum { STATE_START, STATE_END, SUBSTATE_START, SUBSTATE_END,
               BURST_START, BURST_END, PTP_START, PTP_END, POINT, EVENT_TYPES } nanos_event_type_t; /**< Event types  */

typedef enum { NOT_CREATED, NOT_TRACED, STARTUP, SHUTDOWN, ERROR, IDLE, RUNTIME, RUNNING, SYNCHRONIZATION,
               SCHEDULING, CREATION, MEM_TRANSFER, CACHE, YIELD, EVENT_STATE_TYPES
} nanos_event_state_value_t; /**< State enum values */

typedef enum { NANOS_WD_DOMAIN } nanos_event_domain_t; /**< Specifies a domain */
typedef unsigned int  nanos_event_id_t;                /**< Used as unique id within a given domain */

typedef unsigned int         nanos_event_key_t;   /**< Key (on key-value pair) */
typedef unsigned long long   nanos_event_value_t; /**< Value (on key-value pair) */
  
typedef struct {
   nanos_event_key_t    key;
   nanos_event_value_t  value;
} nanos_event_burst_t;

typedef struct {
   nanos_event_state_value_t value;
} nanos_event_state_t;

typedef struct {
   unsigned int        nkvs;
   nanos_event_key_t   *keys;
   nanos_event_value_t *values;
} nanos_event_point_t;

typedef struct {
   nanos_event_domain_t domain; 
   nanos_event_id_t     id;
   unsigned int         nkvs;
   nanos_event_key_t    *keys;
   nanos_event_value_t  *values;
} nanos_event_ptp_t;

typedef struct {
   nanos_event_type_t       type;
   union {
      nanos_event_burst_t   burst;
      nanos_event_state_t   state;
      nanos_event_point_t   point;
      nanos_event_ptp_t     ptp;
   } info;
} nanos_event_t;


#endif
