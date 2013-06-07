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
#include <stdbool.h>

/*! \file nanos_c_api_macros.h
 *  \brief 
 */
#define NANOS_API_DECL(Type, Name, Params) \
    extern Type Name##_ Params; \
    extern Type Name Params

#ifdef _NANOS_INTERNAL

   #define NANOS_API_DEF(Type, Name, Params) \
       __attribute__((alias(#Name))) Type Name##_ Params; \
       Type Name Params

#endif

/*! \defgroup capi C/C++ API */
/*! \addtogroup capi
 *  \{
 */

#ifdef __cplusplus
extern "C"
#endif
typedef struct {
   /* NOTE: The first dimension is represented in terms of bytes. */

   /* Size of the dimension in terms of the size of the previous dimension. */
   size_t size;

   /* Lower bound in terms of the size of the previous dimension. */
   size_t lower_bound;
   /* Accessed length in terms of the size of the previous dimension. */
   size_t accessed_length;
} nanos_region_dimension_internal_t;

#ifdef __cplusplus
extern "C"
#endif
typedef struct {
   bool  input: 1;
   bool  output: 1;
   bool  can_rename:1;
   bool  concurrent: 1;
   bool  commutative: 1;
} nanos_access_type_internal_t;

/* This structure is initialized in dataaccess.hpp. Any change in
 * its contents has to be reflected in DataAccess constructor
 */
#ifdef __cplusplus
extern "C"
#endif
typedef struct {
   /* Base address of the accessed range */
   void *address;
   
   nanos_access_type_internal_t flags;
   
   /* Number of dimensions */
   short dimension_count;
   
   /* The first dimension will be the contiguous one, and its size and
    * offset must be expressed in bytes, not elements.
    */
#if defined(_MERCURIUM) && defined(_MF03)
   // Fortran makes a strong separation between pointers and arrays and they
   // cannot be mixed in any way. To the eyes of Mercurium the original
   // declaration would be a pointer to a scalar, not a pointer to an array
   void* dimensions;
#else
   nanos_region_dimension_internal_t const *dimensions;
#endif
   
   /* Offset of the first element */
   ptrdiff_t offset;
} nanos_data_access_internal_t;

typedef enum {
   NANOS_PRIVATE,
   NANOS_SHARED,
} nanos_sharing_t;

typedef struct {
   void *original;
   void *privates;
   size_t element_size;
   size_t num_scalars;
   void *descriptor; // This is only used in Fortran, it holds a Fortran array descriptor
   void (*bop)( void *, void *, int num_scalars);
   void (*vop)( int n, void *, void *);
   void (*cleanup)(void *);
} nanos_reduction_t;

/* This structure is initialized in copydata.hpp. Any change in
 * its contents has to be reflected in CopyData constructor
 */
typedef struct {
   void *address;
   nanos_sharing_t sharing;
   struct {
      bool input: 1;
      bool output: 1;
   } flags;
   short dimension_count;

#if defined(_MERCURIUM) && defined(_MF03)
   // Fortran makes a strong separation between pointers and arrays and they
   // cannot be mixed in any way. To the eyes of Mercurium the original
   // declaration would be a pointer to a scalar, not a pointer to an array
   void* dimensions;
#else
   nanos_region_dimension_internal_t const *dimensions;
#endif
   ptrdiff_t offset;
} nanos_copy_data_internal_t;

typedef nanos_access_type_internal_t nanos_access_type_t;
typedef nanos_region_dimension_internal_t nanos_region_dimension_t;

#ifndef _NANOS_INTERNAL

typedef nanos_data_access_internal_t nanos_data_access_t;
typedef nanos_copy_data_internal_t nanos_copy_data_t;

#else

namespace nanos {
   class DataAccess;
   class CopyData;
}
typedef nanos::DataAccess nanos_data_access_t;
typedef nanos::CopyData nanos_copy_data_t;

#endif

// C++ types hidden as void *
typedef void * nanos_thread_t;
typedef void * nanos_wd_t;                                                                                                                               
// SlicerCompoundWD data structure
typedef struct {
   int nsect;
   nanos_wd_t lwd[];
} nanos_compound_wd_data_t;

// SlicerRepeatN data structure
typedef struct {
   int n;
} nanos_repeat_n_info_t;

// SlicerFor data structure
typedef struct {
   int lower;
   int upper;
   int step;
   bool last;
   bool wait;
   int chunk;
   int stride;
   int thid;
   int threads;
   void *args;
} nanos_loop_info_t;

// C++ types hidden as void *
typedef void * nanos_ws_t;      // type for a worksharing plugin
typedef void * nanos_ws_info_t; // type for user provided information describing a worksharing (used in create service)
typedef void * nanos_ws_data_t; // abstract type to specify all data needed for a worksharing construct (used internally to communicate all threads)
typedef void * nanos_ws_item_t; // abstract type to specify a portion for a worksharing construct (used in next item service)

typedef struct {
   int           lower_bound;  // loop lower bound
   int           upper_bound;  // loop upper bound
   int           loop_step;    // loop step
   int           chunk_size;   // loop chunk size
} nanos_ws_info_loop_t; /* nanos_ws_info_t, specific loop data */

typedef struct {
   int   lower;      // loop item lower bound
   int   upper;      // loop item upper bound
   bool  execute:1;  // is a valid loop item?
   bool  last:1;     // is the last loop item?
} nanos_ws_item_loop_t; /* nanos_ws_item_t, specific loop data */

typedef struct nanos_ws_desc {
   volatile nanos_ws_t   ws;         // Worksharing plugin (specified at worksharing create service), API -> Worksharing plugin
   nanos_ws_data_t       data;       // Worksharing plugin data (specified at worksharing create service), API -> Worksharing plugin
   struct nanos_ws_desc *next;       // Sequence management: this is 'next' global enqueued worksharing descriptor, internal use
   nanos_thread_t       *threads;    // Slicer plugin information: supporting thread map (lives at slicer creator's stack), API -> Slicer plugin
   int                   nths;       // Slicer plugin information: number of supporting threads, API ->  Slicer plugin
} nanos_ws_desc_t;

// WD const properties
typedef struct {
   bool mandatory_creation:1;
   bool tied:1;
   bool clear_chunk:1;
   bool reserved0:1;
   bool reserved1:1;
   bool reserved2:1;
   bool reserved3:1;
   bool reserved4:1;
} nanos_wd_props_t;

typedef struct {
   bool is_final:1;
   bool reserved1:1;
   bool reserved2:1;
   bool reserved3:1;
   bool reserved4:1;
   bool reserved5:1;
   bool reserved6:1;
   bool reserved7:1;
} nanos_wd_dyn_flags_t;

typedef struct {
   nanos_wd_dyn_flags_t flags;
   nanos_thread_t tie_to;
   unsigned int priority;
} nanos_wd_dyn_props_t;

typedef struct {
  void * (*factory) (void *arg);
  //size_t dd_size;
  void * arg;
} nanos_device_t;

/*! \todo Move nanos_smp_args_t to some dependent part ? */
typedef struct {
   void (*outline) (void *);
} nanos_smp_args_t;

#ifdef __cplusplus
extern "C" {
#endif
// factories
// smp
NANOS_API_DECL(void *, nanos_smp_factory,( void *args));
#define NANOS_SMP_DESC( args ) { nanos_smp_factory, &( args ) }

#ifdef __cplusplus
};
#endif
// instrumentation structures

typedef enum { NANOS_STATE_START, NANOS_STATE_END, NANOS_SUBSTATE_START, NANOS_SUBSTATE_END,
               NANOS_BURST_START, NANOS_BURST_END, NANOS_PTP_START, NANOS_PTP_END, NANOS_POINT, EVENT_TYPES
} nanos_event_type_t; /**< Event types  */

typedef unsigned int         nanos_event_key_t; /**< Key (on key-value pair) */
typedef unsigned long long   nanos_event_value_t; /**< Value (on key-value pair) */

typedef enum { NANOS_NOT_CREATED, NANOS_NOT_RUNNING, NANOS_STARTUP, NANOS_SHUTDOWN, NANOS_ERROR, NANOS_IDLE,
               NANOS_RUNTIME, NANOS_RUNNING, NANOS_SYNCHRONIZATION, NANOS_SCHEDULING, NANOS_CREATION,
               NANOS_MEM_TRANSFER_IN, NANOS_MEM_TRANSFER_OUT, NANOS_MEM_TRANSFER_LOCAL,
               NANOS_MEM_TRANSFER_DEVICE_IN, NANOS_MEM_TRANSFER_DEVICE_OUT, NANOS_MEM_TRANSFER_DEVICE_LOCAL,
               NANOS_CACHE, NANOS_YIELD, NANOS_ACQUIRING_LOCK, NANOS_CONTEXT_SWITCH, NANOS_DEBUG, NANOS_EVENT_STATE_TYPES
} nanos_event_state_value_t; /**< State enum values */

typedef enum { NANOS_WD_DOMAIN, NANOS_WD_DEPENDENCY, NANOS_WAIT, NANOS_WD_REMOTE, NANOS_XFER_PUT, NANOS_XFER_GET
} nanos_event_domain_t; /**< Specifies a domain */

typedef long long  nanos_event_id_t; /**< Used as unique id within a given domain */
  
typedef struct {
   nanos_event_type_t   type;
   nanos_event_key_t    key;
   nanos_event_value_t  value;
   nanos_event_domain_t domain; 
   nanos_event_id_t     id;
} nanos_event_t;

/* Lock C interface */
typedef enum { NANOS_LOCK_FREE = 0, NANOS_LOCK_BUSY = 1 } nanos_lock_state_t;
typedef struct nanos_lock_t {
   volatile nanos_lock_state_t state_;
#ifdef __cplusplus
   nanos_lock_t ( nanos_lock_state_t init=NANOS_LOCK_FREE ) : state_(init) {}
#endif
} nanos_lock_t;

/* Translation function type  */
typedef void (* nanos_translate_args_t) (void *, nanos_wd_t);

/* This types are for the symbols in the linker section for function initialization */
typedef void (nanos_init_func_t) ( void * );
typedef struct {
   nanos_init_func_t  *func;
   void               *data;
} nanos_init_desc_t;

/*!
 * \}
 */ 

#endif
