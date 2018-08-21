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

#include "openclconfig.hpp"
#include "openclplugin.hpp"
#include "system.hpp"
#include <dlfcn.h>

// This symbol is used to detect that a specific feature of OmpSs is used in an application
// (i.e. Mercurium explicitly defines this symbol when a OCL task is defined)
extern "C"
{
   __attribute__((weak)) void nanos_needs_opencl_fun(void);
}

using namespace nanos;
using namespace nanos::ext;

bool OpenCLConfig::_enableOpenCL = false;
bool OpenCLConfig::_forceDisableOpenCL = false;
bool OpenCLConfig::_disableAllocWide = false;
bool OpenCLConfig::_disableOCLdev2dev = false;
size_t OpenCLConfig::_devCacheSize = 0;
bool OpenCLConfig::_forceShMem = false;
int OpenCLConfig::_devNum = INT_MAX;
int OpenCLConfig::_prefetchNum = 1;
unsigned int OpenCLConfig::_currNumDevices = 0;
System::CachePolicyType OpenCLConfig::_cachePolicy = System::DEFAULT;
//System::CachePolicyType OpenCLConfig::_cachePolicy = System::WRITE_BACK;
bool OpenCLConfig::_enableProfiling = false;

std::map<cl_device_id,cl_context>* OpenCLConfig::_devicesPtr=0;

Atomic<unsigned> OpenCLConfig::_freeDevice = 0;

cl_device_id OpenCLConfig::getFreeDevice() {
   if(_freeDevice == _devicesPtr->size())
      fatal( "No more free devices" );
   
   int freeDev=_freeDevice++;
   
   std::map<cl_device_id,cl_context>::iterator iter=_devicesPtr->begin();
   for (int i=0; i < freeDev; ++i){
       ++iter;
   }

   return iter->first;
}

cl_context OpenCLConfig::getContextDevice(cl_device_id dev) {
   return (*_devicesPtr)[dev];
}

bool OpenCLConfig::getAllocWide() {
   return !_disableAllocWide;
}

void OpenCLConfig::prepare( Config &cfg )
{

   // Enable/disable OpenCL.
   cfg.registerConfigOption( "enable-opencl",
                             NEW Config::FlagOption( _enableOpenCL ),
                             "Enable the use of "
                             "OpenCL back-end" );
   cfg.registerEnvOption( "enable-opencl", "NX_ENABLEOPENCL" );
   cfg.registerArgOption( "enable-opencl", "enable-opencl" );
   
   // Enable/disable OpenCL.
   cfg.registerConfigOption( "disable-opencl",
                             NEW Config::FlagOption( _forceDisableOpenCL ),
                             "Disable the use of "
                             "OpenCL back-end" );
   cfg.registerEnvOption( "disable-opencl", "NX_DISABLEOPENCL" );
   cfg.registerArgOption( "disable-opencl", "disable-opencl" );

    // Select the size of the device cache.
   cfg.registerConfigOption( "opencl-cache",
                             NEW Config::SizeVar( _devCacheSize ),
                             "Defines the amount of the cache "
                             "to be allocated on the device (bytes). "
                             " If this number is below 100, the amount of memory is taken as a percentage of the total device memory") ;
   cfg.registerEnvOption( "opencl-cache", "NX_OPENCL_CACHE" );
   cfg.registerArgOption( "opencl-cache", "opencl-cache" );
   
    // Select the size of the device cache.
   cfg.registerConfigOption( "opencl-num-prefetch",
                             NEW Config::IntegerVar( _prefetchNum ),
                             "Defines the maximum number of OpenCL tasks to prefetch (defaults to 1) ");
   cfg.registerEnvOption( "opencl-num-prefetch", "NX_OPENCL_NUM_PREFETCH" );
   cfg.registerArgOption( "opencl-num-prefetch", "opencl-num-prefetch" );
   
       // Select the size of the device cache.
   cfg.registerConfigOption( "opencl-max-devices",
                             NEW Config::IntegerVar( _devNum ),
                             "Defines the total maximum number of devices "
                             "to be used by nanox" );
   cfg.registerEnvOption( "opencl-max-devices", "NX_OPENCL_MAX_DEVICES" );
   cfg.registerArgOption( "opencl-max-devices", "opencl-max-devices" );

   cfg.registerConfigOption( "opencl-alloc-wide", NEW Config::FlagOption( _disableAllocWide ),
                                "Do not alloc full objects in the cache." );
   cfg.registerEnvOption( "opencl-alloc-wide", "NX_OPENCL_DISABLE_ALLOCWIDE" );
   cfg.registerArgOption( "opencl-alloc-wide", "opencl-disable-alloc-wide" );
   
   cfg.registerConfigOption( "opencl-disable-devtodev", NEW Config::FlagOption( _disableOCLdev2dev ),
                                "Disable OpenCL dev to dev." );
   cfg.registerEnvOption( "opencl-disable-devtodev", "NX_OPENCL_DISABLE_DEVTODEV" );
   cfg.registerArgOption( "opencl-disable-devtodev", "opencl-disable-devtodev" );
   
   cfg.registerConfigOption( "force-opencl-mapped",
                             NEW Config::FlagOption( _forceShMem ),
                             "Force the use the use of mapped pointers for every device (Default: GPU -> NO, CPU->YES). Can save copy time on shared memory devices" );
   cfg.registerEnvOption( "force-opencl-mapped", "NX_FORCE_OPENCL_MAPPED");
   cfg.registerArgOption( "force-opencl-mapped", "force-opencl-mapped" );
   
   System::CachePolicyConfig *cachePolicyCfg = NEW System::CachePolicyConfig ( _cachePolicy );
   cachePolicyCfg->addOption("wt", System::WRITE_THROUGH );
   cachePolicyCfg->addOption("wb", System::WRITE_BACK );
   cachePolicyCfg->addOption( "nocache", System::NONE );
   cfg.registerConfigOption ( "opencl-cache-policy", cachePolicyCfg, "Defines the cache policy for OpenCL architectures: write-through / write-back (wb by default)" );
   cfg.registerEnvOption ( "opencl-cache-policy", "NX_OPENCL_CACHE_POLICY" );
   cfg.registerArgOption( "opencl-cache-policy", "opencl-cache-policy" );

   // Enable/disable Profiling.
   const std::string openclProfilingOptName = "opencl-profiling";
   cfg.registerConfigOption( openclProfilingOptName,
                             NEW Config::FlagOption( _enableProfiling ),
                             "Enable the OpenCL Profiling mode");
   cfg.registerEnvOption( openclProfilingOptName, "NX_OPENCL_PROFILING" );
   cfg.registerArgOption( openclProfilingOptName, openclProfilingOptName );
}

void OpenCLConfig::apply(const std::string devTypeIn, std::map<cl_device_id, cl_context>* devices) {
    std::string devTyStr = devTypeIn;
    _devicesPtr = devices;

    void * myself = dlopen(NULL, RTLD_LAZY | RTLD_GLOBAL);

   //For more information see  #1214
   bool mercurium_has_tasks = nanos_needs_opencl_fun;

    dlclose( myself );

    //Auto-enable CUDA if it was not done before
    if (!_enableOpenCL) {
        _enableOpenCL = mercurium_has_tasks;
    }
    if (_forceDisableOpenCL || !_enableOpenCL)
        return;

    cl_int errCode;

    // Get the number of available platforms.
    cl_uint numPlats;
    if (clGetPlatformIDs(0, NULL, &numPlats) != CL_SUCCESS)
        fatal0("Cannot detect the number of available OpenCL platforms");

    if (numPlats == 0)
        fatal0("No OpenCL platform available");

    // Read all platforms.
    cl_platform_id *plats = new cl_platform_id[numPlats];
    if (clGetPlatformIDs(numPlats, plats, NULL) != CL_SUCCESS)
        fatal0("Cannot load OpenCL platforms");

    // Is platform available?
    if (!numPlats)
        fatal0("No OpenCL platform available");

    std::vector<cl_platform_id> _plats;
    // Save platforms.
    _plats.assign(plats, plats + numPlats);
    delete [] plats;

    cl_device_type devTy=0;

    std::transform(devTyStr.begin(), devTyStr.end(), devTyStr.begin(), ::toupper);
    // Parse the requested device type.
    if (devTyStr == "" || devTyStr.find("ALL") != std::string::npos)
            devTy = CL_DEVICE_TYPE_ALL;
    else {
        if (devTyStr.find("CPU") != std::string::npos)
            devTy |= CL_DEVICE_TYPE_CPU;
        if (devTyStr.find("GPU") != std::string::npos)
            devTy |= CL_DEVICE_TYPE_GPU;
        if (devTyStr.find("ACCELERATOR") != std::string::npos)
            devTy |= CL_DEVICE_TYPE_ACCELERATOR;
    }

    // Read all devices.
    for (std::vector<cl_platform_id>::iterator i = _plats.begin(),
            e = _plats.end();
            i != e;
            ++i) {
#ifdef NANOS_ENABLE_ALLOCATOR
        char buffer[200];
        clGetPlatformInfo(*i, CL_PLATFORM_VENDOR, 200, buffer, NULL);
        if (std::string(buffer) == "Intel(R) Corporation" || std::string(buffer) == "ARM") {
            debug0("Intel or ARM OpenCL don't work correctly when using nanox allocator, "
                    "please configure and reinstall nanox with --disable-allocator in case you want to use it, skipping Intel OpenCL devices");
            continue;
        }
#endif
        // Get the number of available devices.
        cl_uint numDevices;
        errCode = clGetDeviceIDs(*i, devTy, 0, NULL, &numDevices);

        if (errCode != CL_SUCCESS)
            continue;

        // Read all matching devices.
        cl_device_id *devs = new cl_device_id[numDevices];
        errCode = clGetDeviceIDs(*i, devTy, numDevices, devs, NULL);
        if (errCode != CL_SUCCESS)
            continue;

        int devicesToUse = 0;
        cl_device_id *avaiableDevs = new cl_device_id[numDevices];
        // Get all avaiable devices
        for (cl_device_id *j = devs, *f = devs + numDevices; j != f; ++j) {
            cl_bool available;

            errCode = clGetDeviceInfo(*j,
                    CL_DEVICE_AVAILABLE,
                    sizeof ( cl_bool),
                    &available,
                    NULL);
            if (errCode != CL_SUCCESS)
                continue;

            unsigned int maxDevs= (unsigned int) _devNum;
            if (available && _devicesPtr->size() + devicesToUse < maxDevs) {
                avaiableDevs[devicesToUse++] = *j;
            }
        }

        cl_context_properties props[] ={CL_CONTEXT_PLATFORM,
            reinterpret_cast<cl_context_properties> (*i),
            0};

        //Cant instrument here
        //NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_CREATE_CONTEXT_EVENT );
        cl_context ctx = clCreateContext(props, devicesToUse, avaiableDevs, NULL, NULL, &errCode);
        //NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
        // Put all available devices inside the vector.
        for (cl_device_id *j = avaiableDevs, *f = avaiableDevs + devicesToUse; j != f; ++j) {
            _devicesPtr->insert(std::make_pair(*j, ctx));
        }

        delete [] devs;
    }   
    _currNumDevices = _devicesPtr->size();
    
    if ( _currNumDevices == 0 ) {
       if ( mercurium_has_tasks ) {
          message0( " OpenCL tasks were compiled and no OpenCL devices were found, execution"
                  " could have unexpected behavior and can even hang" );
       } else {
           message0( " OpenCL plugin was enabled and no OpenCL devices were found " );
       }
    }
}
