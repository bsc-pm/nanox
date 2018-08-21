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

#ifndef _NANOS_OpenCL_CFG
#define _NANOS_OpenCL_CFG

#ifdef HAVE_OPENCL_OPENCL_H
#include <OpenCL/opencl.h>
#endif

#ifdef HAVE_CL_OPENCL_H
#include <CL/opencl.h>
#endif

#include "atomic.hpp"
#include "config.hpp"

namespace nanos {
namespace ext {

class OpenCLPlugin;

class OpenCLConfig {
public:
  OpenCLConfig() {}
  ~OpenCLConfig() {}
  static unsigned getOpenCLDevicesCount() { return _currNumDevices; }
  static cl_device_id getFreeDevice();
  static cl_context getContextDevice(cl_device_id dev);
  static bool getAllocWide();
  static bool getDisableDev2Dev() { return _disableOCLdev2dev; }
  static size_t getDevCacheSize() { return _devCacheSize; }
  static bool getForceShMem() { return _forceShMem; } 
  static int getPrefetchNum() { return _prefetchNum +1; }
  static System::CachePolicyType getCachePolicy ( void ) { return _cachePolicy; }
  static bool isEnableProfiling() { return _enableProfiling; }

private:
  static void prepare( Config &cfg );
  static void apply( const std::string devTypeIn, std::map<cl_device_id, cl_context>* devices );

private:
  // These properties contains raw info set by the user.

  // Whether to disable OpenCL.
  static bool _enableOpenCL;
  
  // Whether to disable OpenCL.
  static bool _forceDisableOpenCL;
  
  // The platform to use.
  static std::string _platName;

  // The portion of the cache to be allocated on the device.
  static size_t _devCacheSize;
  static bool _forceShMem;
  
  //Maximum number of devices to be used by nanox
  static int _devNum;
  //Number of prefetchs
  static int _prefetchNum;
  // These properties contains runtime info, not directly settable by the user.

  // All found OpenCL platforms.
  //static std::vector<cl_platform_id> _plats;
  static std::map<cl_device_id, cl_context>* _devicesPtr;
  static unsigned int _currNumDevices;
  static bool _disableAllocWide; //! Use wide allocation policy for the region cache

  // These properties manages mutable state.

  // The next free device.
  static Atomic<unsigned> _freeDevice;
  // Whether to disable OpenCL dev2dev.
  static bool _disableOCLdev2dev;
  static System::CachePolicyType   _cachePolicy; //! Defines the cache policy used by OCL devices

  // Whether to enable profiling: default=false
  static bool _enableProfiling;

  friend class OpenCLPlugin;
};


// Macro's to instrument the code and make it cleaner
#define NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT(x)   NANOS_INSTRUMENT( \
		sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "in-opencl-runtime" ), (x) ); )

#define NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT       NANOS_INSTRUMENT( \
		sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "in-opencl-runtime" ), 0 ); )


typedef enum {
   NANOS_OPENCL_NULL_EVENT,                            /* 0 */
   NANOS_OPENCL_ALLOC_EVENT,                          /* 1 */
   NANOS_OPENCL_FREE_EVENT,                            /* 2 */
   NANOS_OPENCL_GET_DEV_INFO_EVENT,                     /* 3 */
   NANOS_OPENCL_CREATE_CONTEXT_EVENT,                       /* 4 */
   NANOS_OPENCL_MEMWRITE_SYNC_EVENT,                         /* 5 */
   NANOS_OPENCL_MEMREAD_SYNC_EVENT,                 /* 6 */
   NANOS_OPENCL_CREATE_COMMAND_QUEUE_EVENT,                   /* 7 */
   NANOS_OPENCL_GET_PROGRAM_EVENT,                   /* 8 */
   NANOS_OPENCL_COPY_BUFFER_EVENT,                   /* 9 */
   NANOS_OPENCL_CREATE_SUBBUFFER_EVENT,                   /* 10 */
   NANOS_OPENCL_MAP_BUFFER_SYNC_EVENT,                 /* 11 */
   NANOS_OPENCL_UNMAP_BUFFER_SYNC_EVENT,                 /* 12 */
   NANOS_OPENCL_GENERIC_EVENT,                         /* 13 */
   NANOS_OPENCL_PROFILE_KERNEL,                         /* 14 */
   NANOS_OPENCL_PROFILE_UPDATE_DATA,                     /* 15 */
   NANOS_OPENCL_PROFILE_DB_ACCESS                        /* 16 */
} in_opencl_runtime_event_value;

} // namespace ext
} // namespace nanos

#endif // _NANOS_OpenCL_CFG
