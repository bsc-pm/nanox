
#include "openclprocessor.hpp"
#include "openclthread.hpp"
#include "os.hpp"
#include <iostream>

using namespace nanos;
using namespace nanos::ext;

//
// OpenCLAdapter implementation.
//

OpenCLAdapter::~OpenCLAdapter()
{
  cl_int errCode;

  errCode = clReleaseCommandQueue( _queue );
  if( errCode != CL_SUCCESS )
     fatal0( "Unable to release the command queue" );

  errCode = clReleaseContext( _ctx );
  if( errCode != CL_SUCCESS )
     fatal0( "Unable to release the context" );

  for( ProgramCache::iterator i = _progCache.begin(),
                              e = _progCache.end();
                              i != e;
                              ++i )
    clReleaseProgram( i->second );
}

void OpenCLAdapter::initialize(cl_device_id dev)
{
   cl_int errCode;

   _dev = dev;

   // Get device platform.   
   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_GET_DEV_INFO_EVENT );
   cl_platform_id plat;
   errCode = clGetDeviceInfo( dev,
                                CL_DEVICE_PLATFORM,
                                sizeof(cl_platform_id),
                                &plat,
                                NULL );
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
   if( errCode != CL_SUCCESS )
      fatal0( "Cannot get device platform" );

   // Setup context creation properties.
   cl_context_properties props[] =
      {  CL_CONTEXT_PLATFORM,
         reinterpret_cast<cl_context_properties>(plat),
         0
      };

   // Create the context.
   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_CREATE_CONTEXT_EVENT );
   _ctx = clCreateContext( props, 1, &_dev, NULL, NULL, &errCode );   
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
   
   if( errCode != CL_SUCCESS )
      fatal0( "Cannot create the context" );

   // Get a command queue.
   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_CREATE_COMMAND_QUEUE_EVENT );
   _queue = clCreateCommandQueue( _ctx, _dev, 0, &errCode );
   if( errCode != CL_SUCCESS )
     fatal0( "Cannot create a command queue" );
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
}

cl_int OpenCLAdapter::allocBuffer( size_t size, cl_mem &buf )
{
   cl_int errCode;

   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_ALLOC_EVENT );
   buf = clCreateBuffer( _ctx, CL_MEM_READ_WRITE, size, NULL, &errCode );
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;

   return errCode;
}

cl_int OpenCLAdapter::freeBuffer( cl_mem &buf )
{
   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_FREE_EVENT );
   cl_int ret= clReleaseMemObject( buf );   
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
   return ret;
}

cl_int OpenCLAdapter::readBuffer( cl_mem buf,
                               void *dst,
                               size_t offset,
                               size_t size )
{
   cl_int errCode, exitStatus;
   cl_event ev;

   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_MEMREAD_SYNC_EVENT );
   errCode = clEnqueueReadBuffer( _queue,
                                    buf,
                                    true,
                                    offset,
                                    size,
                                    dst,
                                    0,
                                    NULL,
                                    &ev
                                  );
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
   
   if( errCode != CL_SUCCESS )
      return errCode;

   errCode = clGetEventInfo( ev,
                               CL_EVENT_COMMAND_EXECUTION_STATUS,
                               sizeof(cl_int),
                               &exitStatus,
                               NULL
                             );   

   clReleaseEvent( ev );

   return errCode;
}

cl_int OpenCLAdapter::writeBuffer( cl_mem buf,
                                void *src,
                                size_t offset,
                                size_t size )
{
   cl_int errCode, exitStatus;
   cl_event ev;

   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_MEMWRITE_SYNC_EVENT );
   errCode = clEnqueueWriteBuffer( _queue,
                                     buf,
                                     true,
                                     offset,
                                     size,
                                     src,
                                     0,
                                     NULL,
                                     &ev
                                   );
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
   if( errCode != CL_SUCCESS ){
      return errCode;
   }

   errCode = clGetEventInfo( ev,
                               CL_EVENT_COMMAND_EXECUTION_STATUS,
                               sizeof(cl_int),
                               &exitStatus,
                               NULL
                             );
   
   clReleaseEvent( ev );

   return errCode;
}

cl_int OpenCLAdapter::buildProgram( const char *src,
                                 const char *compilerOpts,
                                 cl_program &prog )
{
   cl_int errCode;

   prog = clCreateProgramWithSource( _ctx, 1, &src, NULL, &errCode );
   if( errCode != CL_SUCCESS )
      return errCode;

   errCode = clBuildProgram( prog, 1, &_dev, compilerOpts, NULL, NULL );
   if( errCode != CL_SUCCESS ){
     // Determine the size of the log
     size_t log_size;
     clGetProgramBuildInfo(prog, _dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

     // Allocate memory for the log
     char *log = (char *) malloc(log_size+1);
     // Get the log
     clGetProgramBuildInfo(prog, _dev, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
     log[log_size] = '\0';

     std::stringstream sstm;

     sstm << "OpenCL kernel compilation failed, errors:" << std::endl << log << std::endl;
     // Print the log
     fatal(sstm.str());
      // No matter if this call fails, the relevant error code is the first.
      clReleaseProgram( prog );
   }
   return errCode;
}

cl_int OpenCLAdapter::destroyProgram( cl_program &prog )
{
  return clReleaseProgram( prog );
}

// TODO: use a fixed cache size.
cl_int OpenCLAdapter::putProgram( cl_program &prog )
{
   return clReleaseProgram( prog );
}


void* OpenCLAdapter::createKernel( char* kernel_name, char* ompss_code_file,const char *compilerOpts)        
{
   cl_program prog;
   uint32_t hash = gnuHash( ompss_code_file );
   nanos::ext::OpenCLProcessor *pe=( nanos::ext::OpenCLProcessor * ) getMyThreadSafe()->runningOn();
   ProgramCache progCache = pe->getProgCache();
   if( progCache.count( hash ) )
   {
      prog = progCache[hash];
   } else {    
      char* ompss_code;    
      FILE *fp;
      size_t source_size;
      fp = fopen(ompss_code_file, "r");
      if (!fp) {
        fatal0("Failed to open file when loading kernel from file " + std::string(ompss_code_file) + ".\n");
      }      
      fseek(fp, 0, SEEK_END); // seek to end of file;
      size_t size = ftell(fp); // get current file pointer
      fseek(fp, 0, SEEK_SET); // seek back to beginning of file
      ompss_code = new char[size];
      source_size = fread( ompss_code, 1, size, fp);
      fclose(fp); 
      ompss_code[size]=0;
      
      std::string filename(ompss_code_file);
      if (filename.find(".cl")!=filename.npos){
         buildProgram( ompss_code, compilerOpts, prog );
      } else {
         cl_int errCode;
         const unsigned char* tmp_code=reinterpret_cast<const unsigned char*>(ompss_code);
         prog = clCreateProgramWithBinary( _ctx, 1, &_dev, &size, &tmp_code, NULL, &errCode );    
         if( errCode != CL_SUCCESS )
            fatal0("Failed to create program with binary from file " + std::string(ompss_code_file) + ".\n");
      }
      delete [] ompss_code;
      progCache[hash] = prog;
   }
   
   
   cl_kernel kern;
   cl_int errCode;
   kern = clCreateKernel( prog, kernel_name, &errCode );
   return kern;    
}

cl_int OpenCLAdapter::execKernel(void* oclKernel, 
                        int workDim, 
                        size_t* ndrOffset, 
                        size_t* ndrLocalSize, 
                        size_t* ndrGlobalSize)
{
   cl_kernel openclKernel=(cl_kernel) oclKernel;
   cl_event ev;
   cl_int errCode, exitStatus;

   
   // Exec it.
   errCode = clEnqueueNDRangeKernel( _queue,
                                       openclKernel,
                                       workDim,
                                       ndrOffset,
                                       ndrGlobalSize,
                                       ndrLocalSize,
                                       0,
                                       NULL,
                                       &ev
                                     );
   if( errCode != CL_SUCCESS )
   {
      // Don't worry about exit code, we are cleaning an error.
      clReleaseKernel( openclKernel );
      std::cerr << "Error code" << errCode << "\n";
      fatal0("Error launching OpenCL kernel");
   }

   // Wait for its termination.
   errCode = clWaitForEvents( 1, &ev );
   if( errCode != CL_SUCCESS )
   {
      // Clean up environment.
      clReleaseEvent( ev );
      clReleaseKernel( openclKernel );
      std::cerr << "Error code" << errCode << "\n";
      fatal0("Error launching OpenCL kernel");
   }

   // Check if any errors has occurred.
   errCode = clGetEventInfo( ev,
                               CL_EVENT_COMMAND_EXECUTION_STATUS,
                               sizeof(cl_int),
                               &exitStatus,
                               NULL
                             );
   if( errCode != CL_SUCCESS || exitStatus != CL_SUCCESS )
   {
      // Clean up environment.
      clReleaseEvent( ev );
      clReleaseKernel( openclKernel );
      std::cerr << "Error code" << errCode << "\n";
      fatal0("Error launching OpenCL kernel");
   }

   // Free the event.
   errCode = clReleaseEvent( ev );
   if( errCode != CL_SUCCESS )
   {
      // Clean up environment.
      clReleaseEvent( ev );
      std::cerr << "Error code" << errCode << "\n";
      fatal0("Error launching OpenCL kernel");
   }

   // Free the kernel.
   return clReleaseKernel( openclKernel );
}

// TODO: replace with new APIs.
size_t OpenCLAdapter::getGlobalSize()
{
   cl_int errCode;
   cl_ulong size;

   errCode = clGetDeviceInfo( _dev,
                                CL_DEVICE_GLOBAL_MEM_SIZE,
                                sizeof(cl_ulong),
                                &size,
                                NULL );
   if( errCode != CL_SUCCESS )
      fatal0( "Cannot get device global memory size" );

   return size;
}

cl_int OpenCLAdapter::getSizeTypeMax( unsigned long long &sizeTypeMax )
{
   std::string plat;

   getPlatformName( plat );

   if( plat == "NVIDIA CUDA" )
      return getNVIDIASizeTypeMax( sizeTypeMax );
   else
      return getStandardSizeTypeMax( sizeTypeMax );
}

cl_int
OpenCLAdapter::getPreferredWorkGroupSizeMultiple(
   size_t &preferredWorkGroupSizeMultiple )
{
   // The NVIDIA SDK does not recognize
   // CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE as a valid
   // clGetKernelWorkGroupInfo key -- code must be specialized.

//   std::string plat;
//
//   getPlatformName( plat );

//   if( plat == "NVIDIA CUDA" )
//      return getNVIDIAPreferredWorkGroupSizeMultiple(
//                preferredWorkGroupSizeMultiple );
//   else
    //Function will be different if CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE 
    //is defined or not
      return getStandardPreferredWorkGroupSizeMultiple(
                preferredWorkGroupSizeMultiple );

}

cl_int OpenCLAdapter::getDeviceInfo( cl_device_info key, size_t size, void *value )
{
   return clGetDeviceInfo( _dev, key, size, value, NULL );
}

cl_int
OpenCLAdapter::getStandardPreferredWorkGroupSizeMultiple(
   size_t &preferredWorkGroupSizeMultiple )
{
   #ifdef CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
   // This parameter is architecture-dependent. Unfortunately there is not a
   // corresponding property. Estimate the parameter by using a custom kernel.

   cl_program prog;
   cl_kernel kern;
   cl_int errCode;

   const char *src = "kernel void empty(void) { }";

   errCode = buildProgram( src, prog );
   if( errCode != CL_SUCCESS )
      return CL_OUT_OF_RESOURCES;

   // Create the kernel.
   kern = clCreateKernel( prog, "empty", &errCode );
   if( errCode != CL_SUCCESS )
   {
      destroyProgram( prog );
      return CL_OUT_OF_RESOURCES;
   }

   // Get the parameter to estimate.
   errCode = clGetKernelWorkGroupInfo(
                kern,
                _dev,
                CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                sizeof(size_t),
                &preferredWorkGroupSizeMultiple,
                NULL
              );
   if( errCode != CL_SUCCESS )
   {
      clReleaseKernel( kern );
      destroyProgram( prog );
      return CL_OUT_OF_RESOURCES;
   }

   // Free the kernel.
   errCode = clReleaseKernel( kern );
   if( errCode != CL_SUCCESS )
   {
      destroyProgram( prog );
      return CL_OUT_OF_RESOURCES;
   }

   return destroyProgram( prog );
#else   
  // CL_DEVICE_WARP_SIZE_NV = 0x4003.

  return getDeviceInfo( 0x4003,
                        sizeof( size_t ),
                        &preferredWorkGroupSizeMultiple );
#endif
}

//cl_int
//OpenCLAdapter::getNVIDIAPreferredWorkGroupSizeMultiple(
//   size_t &preferredWorkGroupSizeMultiple )
//{
//  // CL_DEVICE_WARP_SIZE_NV = 0x4003.
//
//  return getDeviceInfo( 0x4003,
//                        sizeof( size_t ),
//                        &preferredWorkGroupSizeMultiple );
//}


cl_int OpenCLAdapter::getStandardSizeTypeMax( unsigned long long &sizeTypeMax )
{
   // This parameter is architecture-dependent. Unfortunately, there is not a
   // corresponding property -- we need an estimate.

   cl_bitfield bits;
   cl_int errCode;

   errCode = getDeviceInfo( CL_DEVICE_ADDRESS_BITS,
                            sizeof( cl_bitfield ),
                            &bits );
   if( errCode != CL_SUCCESS )
      return errCode;

   switch( bits )
   {
   case 32:
      sizeTypeMax = CL_INT_MAX;
      break;

   case 64:
      sizeTypeMax = CL_LONG_MAX;
      break;

   default:
      fatal0( "Cannot get device size type max value" );
   }

   return errCode;
}

cl_int OpenCLAdapter::getNVIDIASizeTypeMax( unsigned long long &sizeTypeMax )
{
   sizeTypeMax = CL_INT_MAX;

   return CL_SUCCESS;
}

cl_int OpenCLAdapter::getPlatformName( std::string &name )
{
   cl_platform_id plat;
   cl_int errCode;

   // Get device platform.
   errCode = getDeviceInfo( CL_DEVICE_PLATFORM, sizeof( cl_platform_id ), &plat );
   if( errCode != CL_SUCCESS )
      return CL_OUT_OF_RESOURCES;

   size_t size;

   // Get platform name size.
   errCode = clGetPlatformInfo( plat, CL_PLATFORM_NAME, 0, NULL, &size );
   if( errCode != CL_SUCCESS )
      return CL_OUT_OF_RESOURCES;

   char *rawName = new char[size];

   // Get platform name.
   errCode = clGetPlatformInfo( plat, CL_PLATFORM_NAME, size, rawName, NULL );
   if( errCode != CL_SUCCESS )
   {
      delete [] rawName;
      return CL_OUT_OF_RESOURCES;
   }

   name = rawName;

   delete [] rawName;
   return errCode;
}

//
// OpenCLProcessor implementation.
//


OpenCLProcessor::OpenCLProcessor( int id, int devId ) :
   CachedAccelerator<OpenCLDevice>( id, &OpenCLDev ),
   _openclAdapter(),
   _cache( _openclAdapter ),
   _dma ( *this ),
   _devId ( devId ) { }


//TODO: Configure cache awareness
void OpenCLProcessor::initialize()
{
   // Initialize the adapter, it will talk with the OpenCL device.
   _openclAdapter.initialize( OpenCLConfig::getFreeDevice() );

   // Initialize the caching subsystem.
   _cache.initialize();

   // Register this device as cache-aware.
   configureCache( _cache.getSize(), OpenCLConfig::getCachePolicy());
}

WD &OpenCLProcessor::getWorkerWD() const
{
    OpenCLDD * dd = NEW OpenCLDD((OpenCLDD::work_fct)Scheduler::workerLoop);
    WD *wd = NEW WD(dd);
    return *wd;
}


WD & OpenCLProcessor::getMasterWD() const {    
   fatal("Attempting to create a OpenCL master thread");
}

BaseThread &OpenCLProcessor::createThread( WorkDescriptor &wd )
{

   OpenCLThread &thr = *NEW OpenCLThread( wd, this );

   return thr;
}

void OpenCLProcessor::setKernelBufferArg(void* openclKernel, int argNum, void* pointer)
{
    cl_mem buffer=_cache.toMemoryObjSS( pointer );
    //Set buffer as arg
    cl_int errCode= clSetKernelArg( (cl_kernel) openclKernel, argNum, sizeof(cl_mem), &buffer ); 
    if( errCode != CL_SUCCESS )
    {
         fatal0("Error setting kernel buffer arg");
    }
}

void OpenCLProcessor::setKernelArg(void* opencl_kernel, int arg_num, size_t size, void* pointer){
    cl_int errCode= clSetKernelArg( (cl_kernel) opencl_kernel, arg_num, size, pointer );
    if( errCode != CL_SUCCESS )
    {
         fatal0("Error setting kernel arg");
    }
}

void OpenCLProcessor::execKernel(void* openclKernel, 
                        int workDim, 
                        size_t* ndrOffset, 
                        size_t* ndrLocalSize, 
                        size_t* ndrGlobalSize){
     cl_int errCode=_openclAdapter.execKernel(openclKernel,
                            workDim,
                            ndrOffset,
                            ndrLocalSize,
                            ndrGlobalSize);
     if( errCode != CL_SUCCESS )
    {
         fatal0("Error executing kernel");
    }
}

static inline std::string bytesToHumanReadable ( size_t bytes )
 {
    double b = bytes;
    int times = 0;
    while ( b / 1024 >= 1.0 ) {
      b /= 1024;
      times++;
    }

    switch ( times ) {
        case 8: // Yottabyte
           return std::string( toString<double>( b ) + " YB" );
           break;
        case 7: // Zettabyte
           return std::string( toString<double>( b ) + " ZB" );
           break;
        case 6: // Exabyte
           return std::string( toString<double>( b ) + " EB" );
           break;
        case 5: // petabyte
           return std::string( toString<double>( b ) + " PB" );
          break;
        case 4: // terabyte
           return std::string( toString<double>( b ) + " TB" );
           break;
        case 3: // gigabyte
           return std::string( toString<double>( b ) + " GB" );
           break;
        case 2: // megabyte
           return std::string( toString<double>( b ) + " MB" );
           break;
        case 1: // kilobyte
           return std::string( toString<double>( b ) + " KB" );
           break;
        case 0: // byte
        default:
           return std::string( toString<double>( b ) + " B" );
           break;
  }
}

void OpenCLProcessor::printStats ()
{
   message("OpenCL dev" << _devId << " TRANSFER STATISTICS");
   message("    Total input transfers: " << bytesToHumanReadable( _cache._bytesIn.value() ) );
   message("    Total output transfers: " << bytesToHumanReadable( _cache._bytesOut.value() ) );
   message("    Total device transfers: " << bytesToHumanReadable( _cache._bytesDevice.value() ) );
}
 

void OpenCLProcessor::cleanUp()
{
   printStats();
}