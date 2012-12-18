
#include "oclprocessor.hpp"
#include "oclthread.hpp"
#include "os.hpp"
#include <iostream>

using namespace nanos;
using namespace nanos::ext;

//
// OCLAdapter implementation.
//

OCLAdapter::~OCLAdapter()
{
  cl_int errCode;

  errCode = p_clReleaseCommandQueue( _queue );
  if( errCode != CL_SUCCESS )
     fatal0( "Unable to release the command queue" );

  errCode = p_clReleaseContext( _ctx );
  if( errCode != CL_SUCCESS )
     fatal0( "Unable to release the context" );

  for( ProgramCache::iterator i = _progCache.begin(),
                              e = _progCache.end();
                              i != e;
                              ++i )
    p_clReleaseProgram( i->second );
}

void OCLAdapter::initialize(cl_device_id dev)
{
   cl_int errCode;

   _dev = dev;

   // Get device platform.
   cl_platform_id plat;
   errCode = p_clGetDeviceInfo( dev,
                                CL_DEVICE_PLATFORM,
                                sizeof(cl_platform_id),
                                &plat,
                                NULL );
   if( errCode != CL_SUCCESS )
      fatal0( "Cannot get device platform" );

   // Setup context creation properties.
   cl_context_properties props[] =
      {  CL_CONTEXT_PLATFORM,
         reinterpret_cast<cl_context_properties>(plat),
         0
      };

   // Create the context.
   _ctx = p_clCreateContext( props, 1, &_dev, NULL, NULL, &errCode );
   if( errCode != CL_SUCCESS )
      fatal0( "Cannot create the context" );

   // Get a command queue.
   _queue = p_clCreateCommandQueue( _ctx, _dev, 0, &errCode );
   if( errCode != CL_SUCCESS )
     fatal0( "Cannot create a command queue" );
}

cl_int OCLAdapter::allocBuffer( size_t size, cl_mem &buf )
{
   cl_int errCode;

   buf = p_clCreateBuffer( _ctx, CL_MEM_READ_WRITE, size, NULL, &errCode );

   return errCode;
}

cl_int OCLAdapter::freeBuffer( cl_mem &buf )
{
   return p_clReleaseMemObject( buf );
}

cl_int OCLAdapter::readBuffer( cl_mem buf,
                               void *dst,
                               size_t offset,
                               size_t size )
{
   cl_int errCode, exitStatus;
   cl_event ev;

   errCode = p_clEnqueueReadBuffer( _queue,
                                    buf,
                                    true,
                                    offset,
                                    size,
                                    dst,
                                    0,
                                    NULL,
                                    &ev
                                  );
   if( errCode != CL_SUCCESS )
      return errCode;

   errCode = p_clGetEventInfo( ev,
                               CL_EVENT_COMMAND_EXECUTION_STATUS,
                               sizeof(cl_int),
                               &exitStatus,
                               NULL
                             );
   if( errCode != CL_SUCCESS || exitStatus != CL_SUCCESS )
      errCode = CL_MEM_OBJECT_ALLOCATION_FAILURE;

   p_clReleaseEvent( ev );

   return errCode;
}

cl_int OCLAdapter::writeBuffer( cl_mem buf,
                                void *src,
                                size_t offset,
                                size_t size )
{
   cl_int errCode, exitStatus;
   cl_event ev;

   errCode = p_clEnqueueWriteBuffer( _queue,
                                     buf,
                                     true,
                                     offset,
                                     size,
                                     src,
                                     0,
                                     NULL,
                                     &ev
                                   );
   if( errCode != CL_SUCCESS ){
      return errCode;
   }

   errCode = p_clGetEventInfo( ev,
                               CL_EVENT_COMMAND_EXECUTION_STATUS,
                               sizeof(cl_int),
                               &exitStatus,
                               NULL
                             );
   
   if( errCode != CL_SUCCESS || exitStatus != CL_SUCCESS )
      errCode = CL_MEM_OBJECT_ALLOCATION_FAILURE;

   p_clReleaseEvent( ev );

   return errCode;
}

cl_int OCLAdapter::buildProgram( const char *src,
                                 const char *compilerOpts,
                                 cl_program &prog )
{
   bool starSSMode=OCLConfig::getStarSSMode();
   cl_int errCode;

   prog = p_clCreateProgramWithSource( _ctx, 1, &src, NULL, &errCode );
   if( errCode != CL_SUCCESS )
      return errCode;

   errCode = p_clBuildProgram( prog, 1, &_dev, compilerOpts, NULL, NULL );
   if( errCode != CL_SUCCESS ){
      //In starSSMode, print compilation otuput
      if (starSSMode){
           // Determine the size of the log
            size_t log_size;
            p_clGetProgramBuildInfo(prog, _dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

            // Allocate memory for the log
            char *log = (char *) malloc(log_size+1);
            // Get the log
            p_clGetProgramBuildInfo(prog, _dev, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            log[log_size] = '\0';
            
            std::stringstream sstm;
            
            sstm << "OpenCL kernel compilation failed, errors:" << std::endl << log << std::endl;
            // Print the log
            fatal(sstm.str());
      }
      // No matter if this call fails, the relevant error code is the first.
      p_clReleaseProgram( prog );
   }
   return errCode;
}

cl_int OCLAdapter::destroyProgram( cl_program &prog )
{
  return p_clReleaseProgram( prog );
}

// TODO: use a fixed cache size.
cl_int OCLAdapter::getProgram( const char *src,
                               const char *compilerOpts,
                               cl_program &prog )
{    
   cl_int errCode;

   uint32_t hash = gnuHash( src );

   if( _progCache.count( hash ) )
   {
      prog = _progCache[hash];
      errCode = p_clRetainProgram( prog );

      return errCode;
   }

       errCode = buildProgram( src, compilerOpts, prog );
   if( errCode != CL_SUCCESS )
       return errCode;

   _progCache[hash] = prog;

   errCode = p_clRetainProgram( prog );
   
   return errCode;
}

// TODO: use a fixed cache size.
cl_int OCLAdapter::putProgram( cl_program &prog )
{
   return p_clReleaseProgram( prog );
}

cl_int OCLAdapter::execKernel( cl_program prog,
                               const char *name,
                               size_t workDim,
                               const size_t *globalWorkOffset,
                               const size_t *globalWorkSize,
                               const size_t *localWorkSize,
                               OCLNDRangeKernelStarSSDD::arg_iterator i,
                               OCLNDRangeKernelStarSSDD::arg_iterator e )
{
   cl_kernel kern;
   cl_event ev;
   cl_int errCode, exitStatus;

   // Create the kernel.
   kern = p_clCreateKernel( prog, name, &errCode );
   if( errCode != CL_SUCCESS )
      return errCode;

   // Set arguments.
   for( unsigned j = 0; i != e; ++i, ++j )
   {
      OCLNDRangeKernelStarSSDD::Arg &arg = *i;

      errCode = p_clSetKernelArg( kern, j, arg._size, arg._ptr );
      if( errCode != CL_SUCCESS ){          
         return errCode;
      }
   }
   
   // Exec it.
   errCode = p_clEnqueueNDRangeKernel( _queue,
                                       kern,
                                       workDim,
                                       globalWorkOffset,
                                       globalWorkSize,
                                       localWorkSize,
                                       0,
                                       NULL,
                                       &ev
                                     );
   if( errCode != CL_SUCCESS )
   {
      // Don't worry about exit code, we are cleaning an error.
      p_clReleaseKernel( kern );
      return errCode;
   }

   // Wait for its termination.
   errCode = p_clWaitForEvents( 1, &ev );
   if( errCode != CL_SUCCESS )
   {
      // Clean up environment.
      p_clReleaseEvent( ev );
      p_clReleaseKernel( kern );
      return errCode;
   }

   // Check if any errors has occurred.
   errCode = p_clGetEventInfo( ev,
                               CL_EVENT_COMMAND_EXECUTION_STATUS,
                               sizeof(cl_int),
                               &exitStatus,
                               NULL
                             );
   if( errCode != CL_SUCCESS || exitStatus != CL_SUCCESS )
   {
      // Clean up environment.
      p_clReleaseEvent( ev );
      p_clReleaseKernel( kern );
      return errCode;
   }

   // Free the event.
   errCode = p_clReleaseEvent( ev );
   if( errCode != CL_SUCCESS )
   {
      // Clean up environment.
      p_clReleaseEvent( ev );
      return errCode;
   }

   // Free the kernel.
   return p_clReleaseKernel( kern );
}

// TODO: replace with new APIs.
size_t OCLAdapter::getGlobalSize()
{
   cl_int errCode;
   cl_ulong size;

   errCode = p_clGetDeviceInfo( _dev,
                                CL_DEVICE_GLOBAL_MEM_SIZE,
                                sizeof(cl_ulong),
                                &size,
                                NULL );
   if( errCode != CL_SUCCESS )
      fatal0( "Cannot get device global memory size" );

   return size;
}

cl_int OCLAdapter::getSizeTypeMax( unsigned long long &sizeTypeMax )
{
   std::string plat;

   getPlatformName( plat );

   if( plat == "NVIDIA CUDA" )
      return getNVIDIASizeTypeMax( sizeTypeMax );
   else
      return getStandardSizeTypeMax( sizeTypeMax );
}

cl_int
OCLAdapter::getPreferredWorkGroupSizeMultiple(
   size_t &preferredWorkGroupSizeMultiple )
{
   // The NVIDIA SDK does not recognize
   // CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE as a valid
   // clGetKernelWorkGroupInfo key -- code must be specialized.

   std::string plat;

   getPlatformName( plat );

   if( plat == "NVIDIA CUDA" )
      return getNVIDIAPreferredWorkGroupSizeMultiple(
                preferredWorkGroupSizeMultiple );
   else
      return getStandardPreferredWorkGroupSizeMultiple(
                preferredWorkGroupSizeMultiple );

}

cl_int OCLAdapter::getDeviceInfo( cl_device_info key, size_t size, void *value )
{
   return p_clGetDeviceInfo( _dev, key, size, value, NULL );
}

cl_int
OCLAdapter::getStandardPreferredWorkGroupSizeMultiple(
   size_t &preferredWorkGroupSizeMultiple )
{
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
   kern = p_clCreateKernel( prog, "empty", &errCode );
   if( errCode != CL_SUCCESS )
   {
      destroyProgram( prog );
      return CL_OUT_OF_RESOURCES;
   }

   // Get the parameter to estimate.
   errCode = p_clGetKernelWorkGroupInfo(
                kern,
                _dev,
                CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                sizeof(size_t),
                &preferredWorkGroupSizeMultiple,
                NULL
              );
   if( errCode != CL_SUCCESS )
   {
      p_clReleaseKernel( kern );
      destroyProgram( prog );
      return CL_OUT_OF_RESOURCES;
   }

   // Free the kernel.
   errCode = p_clReleaseKernel( kern );
   if( errCode != CL_SUCCESS )
   {
      destroyProgram( prog );
      return CL_OUT_OF_RESOURCES;
   }

   return destroyProgram( prog );
}

cl_int
OCLAdapter::getNVIDIAPreferredWorkGroupSizeMultiple(
   size_t &preferredWorkGroupSizeMultiple )
{
  // CL_DEVICE_WARP_SIZE_NV = 0x4003.

  return getDeviceInfo( 0x4003,
                        sizeof( size_t ),
                        &preferredWorkGroupSizeMultiple );
}


cl_int OCLAdapter::getStandardSizeTypeMax( unsigned long long &sizeTypeMax )
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

cl_int OCLAdapter::getNVIDIASizeTypeMax( unsigned long long &sizeTypeMax )
{
   sizeTypeMax = CL_INT_MAX;

   return CL_SUCCESS;
}

cl_int OCLAdapter::getPlatformName( std::string &name )
{
   cl_platform_id plat;
   cl_int errCode;

   // Get device platform.
   errCode = getDeviceInfo( CL_DEVICE_PLATFORM, sizeof( cl_platform_id ), &plat );
   if( errCode != CL_SUCCESS )
      return CL_OUT_OF_RESOURCES;

   size_t size;

   // Get platform name size.
   errCode = p_clGetPlatformInfo( plat, CL_PLATFORM_NAME, 0, NULL, &size );
   if( errCode != CL_SUCCESS )
      return CL_OUT_OF_RESOURCES;

   char *rawName = new char[size];

   // Get platform name.
   errCode = p_clGetPlatformInfo( plat, CL_PLATFORM_NAME, size, rawName, NULL );
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
// OCLProcessor implementation.
//

#ifdef CLUSTER_DEV

OCLProcessor::OCLProcessor( int id ) :
   CachedAccelerator<OCLDevice>( id, &OCLDev ),
   _oclAdapter(),
   _cache( _oclAdapter ),
   _dma( *this ),
   _remoteHandler( *this ) { }

#else

OCLProcessor::OCLProcessor( int id ) :
   CachedAccelerator<OCLDevice>( id, &OCLDev ),
   _oclAdapter(),
   _cache( _oclAdapter ),
   _dma ( *this ) { }

#endif // CLUSTER_DEV

void OCLProcessor::initialize()
{
   // Initialize the adapter, it will talk with the OpenCL device.
   _oclAdapter.initialize( OCLConfig::getFreeDevice() );

   // Initialize the caching subsystem.
   _cache.initialize();

   // Register this device as cache-aware.
   configureCache( _cache.getSize(), NANOS_CACHE_WT_POLICY);// toCachePolicy(OCLConfig::getCachePolicy())) ;
}

WD &OCLProcessor::getWorkerWD() const
{
   SMPDD::work_fct work_fct;
   SMPDD *dd;
   WD *wd;

   work_fct = reinterpret_cast<SMPDD::work_fct>( Scheduler::workerLoop );
   dd = NEW SMPDD( work_fct );
   wd = NEW WD( dd );

   return *wd;
}

BaseThread &OCLProcessor::createThread( WorkDescriptor &wd )
{
   OCLLocalThread *thr;

   ensure( wd.canRunIn( SMP ), "Incompatible worker thread" );

   thr = NEW OCLLocalThread( wd, this );

   return *thr;
}


int OCLProcessor::exec( OCLNDRangeKernelStarSSDD &kern,
                        OCLNDRangeKernelStarSSDD::Data &data,
                        OCLNDRangeKernelStarSSDD::arg_iterator i,
                        OCLNDRangeKernelStarSSDD::arg_iterator e )
{

   cl_program prog;
   cl_int errCode;
   
   errCode = _oclAdapter.getProgram( data._programSrcs,
                                     data._compilerOptions,
                                     prog );
   if( errCode != CL_SUCCESS ){
      return errCode;
   }

   bool starSSMode=OCLConfig::getStarSSMode();


   // Before executing the kernel we must translate arguments.
   for( OCLNDRangeKernelStarSSDD::arg_iterator j = i; j != e; ++j )
   {
      OCLNDRangeKernelStarSSDD::Arg &arg = *j;

      // Buffers must be translated into internal representation.
      if( isBufferArg( arg ) )
      {
         // Global buffers.
         if( arg._ptr )
         {
           
            if (starSSMode){                
                cl_mem buffer=_cache.toMemoryObjSS( arg._ptr );
                //If buffer is null, this was not written, but it's probably allocated (output)
                //Search with size and assign it to this pointer
                if (buffer==NULL){
                    arg._size=arg._size-1;
                    buffer=_cache.toMemoryObjSizeSS(arg._size, arg._ptr );
                }
                arg._ptr = new cl_mem( buffer );
            } else {
                unsigned id = getBufferId( arg._size );                
                arg._ptr = new cl_mem( _cache.toMemoryObj( id ) );
            }

            arg._size = sizeof( cl_mem );
            
         }

         // Local buffers.
         else
         {
            arg._size = getLocalBufferSize( arg._size );
         }
      }
   }

   // Exec kernel.
   errCode = _oclAdapter.execKernel( prog,
                                     data._kernName,
                                     kern.getWorkDim(),
                                     kern.getGlobalWorkOffset(),
                                     kern.getGlobalWorkSize(),
                                     kern.getLocalWorkSize(),
                                     i,
                                     e
                                   );

   // Destroy parameters.
   for( OCLNDRangeKernelStarSSDD::arg_iterator j = i; j != e; ++j )
   {
      OCLNDRangeKernelStarSSDD::Arg &arg = *j;

      if( isBufferArg( arg ) && arg._ptr != NULL )
         delete static_cast<cl_mem *>( arg._ptr );
   }

   if( errCode != CL_SUCCESS )
      return errCode;

   return _oclAdapter.putProgram( prog );
}

unsigned long long OCLProcessor::readTicks()
{
   // From caller perspective, here we are inside the device.
   return OS::getRawMonotonicTime();
}
