
#ifndef _NANOS_OpenCL_CFG
#define _NANOS_OpenCL_CFG

#include "atomic.hpp"
#include "openclwrapper.hpp"
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
  static void apply();

private:
  // These properties contains raw info set by the user.

  // Whether to disable OpenCL.
  static bool _disableOpenCL;
  
  // The platform to use.
  static std::string _platName;

  // The device type to consider.
  static std::string _devTy;
  
  //! Defines the cache policy used by OpenCL devices
  static System::CachePolicyType   _cachePolicy; //! Defines the cache policy used by GPU devices


  // The portion of the cache to be allocated on the device.
  static int _devCacheSize;
  
  //Maximum number of devices to be used by nanox
  static unsigned int _devNum;
  // These properties contains runtime info, not directly settable by the user.

  // All found OpenCL platforms.
  static std::vector<cl_platform_id> _plats;

  // All found devices.
  static std::vector<cl_device_id> _devices;

  // These properties manages mutable state.

  // The next free device.
  static Atomic<unsigned> _freeDevice;

  friend class OpenCLPlugin;
};

} // End namespace ext.
} // End namespace nanos.

#endif // _NANOS_OpenCL_CFG
