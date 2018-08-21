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

#ifndef _NANOS_GPU_UTILS
#define _NANOS_GPU_UTILS


#include <iostream>

#include "nanos-int.h"


namespace nanos {
namespace ext {

   class GPUUtils
   {
      public:

         class GPUInstrumentationEventKeys
         {
            public:
               static nanos_event_key_t   _gpu_wd_id;
               static nanos_event_key_t   _in_cuda_runtime;
               static nanos_event_key_t   _user_funct_location;
               static nanos_event_key_t   _copy_in_gpu;
               static nanos_event_key_t   _copy_out_gpu;
         };

         typedef enum {
            NANOS_GPU_CUDA_NULL_EVENT,                            /* 0 */
            NANOS_GPU_CUDA_MALLOC_EVENT,                          /* 1 */
            NANOS_GPU_CUDA_FREE_EVENT,                            /* 2 */
            NANOS_GPU_CUDA_MALLOC_HOST_EVENT,                     /* 3 */
            NANOS_GPU_CUDA_FREE_HOST_EVENT,                       /* 4 */
            NANOS_GPU_CUDA_MEMCOPY_EVENT,                         /* 5 */
            NANOS_GPU_CUDA_MEMCOPY_TO_HOST_EVENT,                 /* 6 */
            NANOS_GPU_CUDA_MEMCOPY_TO_DEVICE_EVENT,               /* 7 */
            NANOS_GPU_CUDA_MEMCOPY_ASYNC_EVENT,                   /* 8 */
            NANOS_GPU_CUDA_MEMCOPY_ASYNC_TO_HOST_EVENT,           /* 9 */
            NANOS_GPU_CUDA_MEMCOPY_ASYNC_TO_DEVICE_EVENT,         /* 10 */
            NANOS_GPU_CUDA_INPUT_STREAM_SYNC_EVENT,               /* 11 */
            NANOS_GPU_CUDA_OUTPUT_STREAM_SYNC_EVENT,              /* 12 */
            NANOS_GPU_CUDA_KERNEL_STREAM_SYNC_EVENT,              /* 13 */
            NANOS_GPU_CUDA_DEVICE_SYNC_EVENT,                     /* 14 */
            NANOS_GPU_CUDA_SET_DEVICE_EVENT,                      /* 15 */
            NANOS_GPU_CUDA_GET_DEVICE_PROPS_EVENT,                /* 16 */
            NANOS_GPU_CUDA_SET_DEVICE_FLAGS_EVENT,                /* 17 */
            NANOS_GPU_CUDA_GET_LAST_ERROR_EVENT,                  /* 18 */
            NANOS_GPU_CUDA_GENERIC_EVENT,                         /* 19 */
            NANOS_GPU_MEMCOPY_EVENT,                              /* 20 */
            NANOS_GPU_CUDA_EVENT_CREATE_EVENT,                    /* 21 */
            NANOS_GPU_CUDA_EVENT_DESTROY_EVENT,                   /* 22 */
            NANOS_GPU_CUDA_EVENT_RECORD_EVENT,                    /* 23 */
            NANOS_GPU_CUDA_EVENT_QUERY_EVENT,                     /* 24 */
            NANOS_GPU_CUDA_EVENT_SYNC_EVENT,                      /* 25 */
            NANOS_GPU_CUDA_KERNEL_LAUNCH_EVENT,                   /* 26 */
            NANOS_GPU_CUDA_STREAM_CREATE_EVENT,                   /* 27 */
            NANOS_GPU_CUDA_STREAM_DESTROY_EVENT,                  /* 28 */
            NANOS_GPU_CUDA_GET_PCI_BUS_EVENT                      /* 29 */
         } GPUInstrumentationInCudaRuntimeEventValue;

         static void displayAllGPUsProperties( void );
         static std::string bytesToHumanReadable ( size_t bytes );

   };


} // namespace ext
} // namespace nanos


// Macro's to instrument the code and make it cleaner
#define NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT(x)   NANOS_INSTRUMENT( \
   sys.getInstrumentation()->raiseOpenBurstEvent ( nanos::ext::GPUUtils::GPUInstrumentationEventKeys::_in_cuda_runtime, (x) ); )

#define NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT       NANOS_INSTRUMENT( \
   sys.getInstrumentation()->raiseCloseBurstEvent ( nanos::ext::GPUUtils::GPUInstrumentationEventKeys::_in_cuda_runtime, 0 ); )



#endif
