
#ifndef _NANOS_OpenCL_CFG
#define _NANOS_OpenCL_CFG

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
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
  static unsigned getOpenCLDevicesCount() { return _devices.size(); }
  static cl_device_id getFreeDevice();

  static size_t getDevCacheSize() { return _devCacheSize; }
  
    
  static System::CachePolicyType getCachePolicy(void) { return _cachePolicy;}

private:
  static void prepare( Config &cfg );
  static void apply(std::string& _devTy);

private:
  // These properties contains raw info set by the user.

  // Whether to disable OpenCL.
  static bool _disableOpenCL;
  
  // The platform to use.
  static std::string _platName;

  
  //! Defines the cache policy used by OpenCL devices
  static System::CachePolicyType   _cachePolicy; //! Defines the cache policy used by GPU devices


  // The portion of the cache to be allocated on the device.
  static int _devCacheSize;
  
  //Maximum number of devices to be used by nanox
  static unsigned int _devNum;
  // These properties contains runtime info, not directly settable by the user.

  // All found OpenCL platforms.
  //static std::vector<cl_platform_id> _plats;

  // All found devices.
  static std::vector<cl_device_id> _devices;

  // These properties manages mutable state.

  // The next free device.
  static Atomic<unsigned> _freeDevice;

  friend class OpenCLPlugin;
};


// Macro's to instrument the code and make it cleaner
#define NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT(x)   NANOS_INSTRUMENT( \
		sys.getInstrumentation()->raiseOpenBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "in-opencl-runtime" ), (x) ); )

#define NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT       NANOS_INSTRUMENT( \
		sys.getInstrumentation()->raiseCloseBurstEvent ( sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey( "in-opencl-runtime" ) ); )


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
   NANOS_OPENCL_GENERIC_EVENT                         /* 9 */
} in_opencl_runtime_event_value;

} // End namespace ext.
} // End namespace nanos.

#endif // _NANOS_OpenCL_CFG
