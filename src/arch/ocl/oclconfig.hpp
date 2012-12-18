
#ifndef _NANOS_OCL_CFG
#define _NANOS_OCL_CFG

#include "atomic.hpp"
#include "oclwrapper.hpp"
#include "config.hpp"

namespace nanos {
namespace ext {

class OCLPlugin;

class OCLConfig {
public:
  OCLConfig() {}
  ~OCLConfig() {}
  static unsigned getOCLDevicesCount() { return _devices.size(); }
  static cl_device_id getFreeDevice();

  static size_t getDevCacheSize() { return _devCacheSize; }
  static size_t getHostCacheSize() { return _hostCacheSize; }
  
  
  static bool getStarSSMode() { return _starSSMode; }
  static void enableStarSSMode() {if (!_starSSMode) _starSSMode=true;};
  
  static std::string getCachePolicy(void) { return _cachePolicy;}

private:
  static void prepare( Config &cfg );
  static void apply();

private:
  // These properties contains raw info set by the user.

  // Whether to disable OpenCL.
  static bool _disableOCL;
  
  // Whether to use StarSS instead of OpenCL (mainlly affects cache), auto-activated when
  // someone uses starSS device
  static bool _starSSMode;

  // The platform to use.
  static std::string _platName;

  // The device type to consider.
  static std::string _devTy;
  
  //! Defines the cache policy used by OpenCL devices
  static std::string _cachePolicy; 


  // The portion of the cache to be allocated on the device.
  static int _devCacheSize;

  // The portion of the cache to be allocated on the host.
  static int _hostCacheSize;

  // These properties contains runtime info, not directly settable by the user.

  // All found OpenCL platforms.
  static std::vector<cl_platform_id> _plats;

  // All found devices.
  static std::vector<cl_device_id> _devices;

  // These properties manages mutable state.

  // The next free device.
  static Atomic<unsigned> _freeDevice;

  friend class OCLPlugin;
};

} // End namespace ext.
} // End namespace nanos.

#endif // _NANOS_OCL_CFG
