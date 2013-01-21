
#include "oclconfig.hpp"
#include "system.hpp"

using namespace nanos;
using namespace nanos::ext;

bool OCLConfig::_disableOCL = false;
std::string OCLConfig::_devTy = "ALL";
int OCLConfig::_devCacheSize = 0;
System::CachePolicyType OCLConfig::_cachePolicy = System::WRITE_BACK;

std::vector<cl_platform_id> OCLConfig::_plats;
std::vector<cl_device_id> OCLConfig::_devices;

Atomic<unsigned> OCLConfig::_freeDevice = 0;

cl_device_id OCLConfig::getFreeDevice() {
   if(_freeDevice == _devices.size())
      fatal( "No more free devices" );

   return _devices[_freeDevice++];
}

void OCLConfig::prepare( Config &cfg )
{
   cfg.setOptionsSection( "OPENCL Arch", "OpenCL specific options" );

   // Enable/disable OpenCL.
   cfg.registerConfigOption( "disable-opencl",
                             NEW Config::FlagOption( _disableOCL ),
                             "Enable or disable the use of "
                             "OpenCL back-end (enabled by default)" );
   cfg.registerEnvOption( "disable-opencl", "NX_DISABLE_OPENCL" );
   cfg.registerArgOption( "disable-opencl", "disable-opencl" );

   // Select the device to use.
   cfg.registerConfigOption( "opencl-device",
                             NEW Config::StringVar( _devTy ),
                             "Defines the OpenCL device type to use "
                             "(ALL, CPU, GPU, ACCELERATOR)" );
   cfg.registerEnvOption( "opencl-device", "NX_OPENCL_DEVICE" );
   cfg.registerArgOption( "opencl-device", "opencl-device" );
   
   System::CachePolicyConfig *cachePolicyCfg = NEW System::CachePolicyConfig ( _cachePolicy );
   cachePolicyCfg->addOption("wt", System::WRITE_THROUGH );
   cachePolicyCfg->addOption("wb", System::WRITE_BACK );
   cachePolicyCfg->addOption( "nocache", System::NONE );
   // Set the cache policy for OpenCL devices
   cfg.registerConfigOption ( "opencl-cache-policy", cachePolicyCfg, "Defines the cache policy for OpenCL devices" );
   cfg.registerEnvOption ( "opencl-cache-policy", "NX_OPENCL_CACHE_POLICY" );
   cfg.registerArgOption( "opencl-cache-policy", "opencl-cache-policy" );

   // Select the size of the device cache.
   cfg.registerConfigOption( "opencl-cache",
                             NEW Config::IntegerVar( _devCacheSize ),
                             "Defines the amount of the cache "
                             "to be allocated on the device" );
   cfg.registerEnvOption( "opencl-cache", "NX_OPENCL_CACHE" );
   cfg.registerArgOption( "opencl-cache", "opencl-cache" );

}

void OCLConfig::apply()
{
   if( _disableOCL )
     return;

   cl_int errCode;

   // Get the number of available platforms.
   cl_uint numPlats;
   if( p_clGetPlatformIDs( 0, NULL, &numPlats ) != CL_SUCCESS )
      fatal0( "Cannot detect the number of available OpenCL platforms" );

   if ( numPlats == 0 )
      fatal0( "No OpenCL platform available" );

   // Read all platforms.
   cl_platform_id *plats = new cl_platform_id[numPlats];
   if( p_clGetPlatformIDs( numPlats, plats, NULL ) != CL_SUCCESS )
      fatal0( "Cannot load OpenCL platforms" );

   // Is platform available?
   if( !numPlats )
      fatal0( "No OpenCL platform available" );

   // Save platforms.
   _plats.assign(plats, plats + numPlats);
   delete [] plats;

   //TODO FIX THIS OR RETURN TO TRUNK MODE
   cl_device_type devTy;

   // Parse the requested device type.
   if( _devTy == "" || _devTy == "ALL" ){
      devTy = CL_DEVICE_TYPE_ALL;
   }  else if( _devTy == "CPU" )
      devTy = CL_DEVICE_TYPE_CPU;
   else if( _devTy == "GPU" )
      devTy = CL_DEVICE_TYPE_GPU;
   else if( _devTy == "ACCELERATOR" )
      devTy = CL_DEVICE_TYPE_ACCELERATOR;
   else
      fatal0( "Unable to parse device type" );

   // Read all devices.
   for( std::vector<cl_platform_id>::iterator i = _plats.begin(),
                                              e = _plats.end();
                                              i != e;
                                              ++i ) {
      // Get the number of available devices.
      cl_uint numDevices;
      errCode = p_clGetDeviceIDs( *i, devTy, 0, NULL, &numDevices );
      if( errCode != CL_SUCCESS )
         continue;

      // Read all matching devices.
      cl_device_id *devs = new cl_device_id[numDevices];
      errCode = p_clGetDeviceIDs( *i, devTy, numDevices, devs, NULL );
      if( errCode != CL_SUCCESS )
         continue;

      // Put all available devices inside the vector.
      for( cl_device_id *j = devs, *f = devs + numDevices; j != f; ++j )
      {
         cl_bool available;

         errCode = p_clGetDeviceInfo( *j,
                                      CL_DEVICE_AVAILABLE,
                                      sizeof( cl_bool ),
                                      &available,
                                      NULL );
         if( errCode != CL_SUCCESS )
           continue;

         if( available )
           _devices.push_back( *j );
      }

      delete [] devs;
   }
}
