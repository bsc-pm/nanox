
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
   cl_int errCode;

   prog = p_clCreateProgramWithSource( _ctx, 1, &src, NULL, &errCode );
   if( errCode != CL_SUCCESS )
      return errCode;

   errCode = p_clBuildProgram( prog, 1, &_dev, compilerOpts, NULL, NULL );
   if( errCode != CL_SUCCESS ){
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
      // No matter if this call fails, the relevant error code is the first.
      p_clReleaseProgram( prog );
   }
   return errCode;
}

cl_int OCLAdapter::destroyProgram( cl_program &prog )
{
  return p_clReleaseProgram( prog );
}

// Get program cache of this processor
void* OCLAdapter::getProgram( const char *src,
                               const char *compilerOpts)
{    
   cl_program prog;

   uint32_t hash = gnuHash( src );
   nanos::ext::OCLProcessor *pe=( nanos::ext::OCLProcessor * ) myThread->runningOn();
   ProgramCache progCache = pe->getProgCache();
   if( progCache.count( hash ) )
   {
      prog = progCache[hash];
   }

   buildProgram( src, compilerOpts, prog );

   progCache[hash] = prog;
   
   return prog;
}

// TODO: use a fixed cache size.
cl_int OCLAdapter::putProgram( cl_program &prog )
{
   return p_clReleaseProgram( prog );
}

void* OCLAdapter::createKernel( char* kernel_name, void* program)        
{
   cl_kernel kern;
   cl_int errCode;
   kern = p_clCreateKernel( (cl_program) program, kernel_name, &errCode );
   return kern;    
}

cl_int OCLAdapter::execKernel(void* openclKernel, 
                        int workDim, 
                        size_t* ndrOffset, 
                        size_t* ndrLocalSize, 
                        size_t* ndrGlobalSize)
{
   cl_kernel oclKernel=(cl_kernel) openclKernel;
   cl_event ev;
   cl_int errCode, exitStatus;

   
   // Exec it.
   errCode = p_clEnqueueNDRangeKernel( _queue,
                                       oclKernel,
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
      p_clReleaseKernel( oclKernel );
      return errCode;
   }

   // Wait for its termination.
   errCode = p_clWaitForEvents( 1, &ev );
   if( errCode != CL_SUCCESS )
   {
      // Clean up environment.
      p_clReleaseEvent( ev );
      p_clReleaseKernel( oclKernel );
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
      p_clReleaseKernel( oclKernel );
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
   return p_clReleaseKernel( oclKernel );
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

cl_int OCLAdapter::getDeviceInfo( cl_device_info key, size_t size, void *value )
{
   return p_clGetDeviceInfo( _dev, key, size, value, NULL );
}

cl_int
OCLAdapter::getStandardPreferredWorkGroupSizeMultiple(
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
#else   
  // CL_DEVICE_WARP_SIZE_NV = 0x4003.

  return getDeviceInfo( 0x4003,
                        sizeof( size_t ),
                        &preferredWorkGroupSizeMultiple );
#endif
}

//cl_int
//OCLAdapter::getNVIDIAPreferredWorkGroupSizeMultiple(
//   size_t &preferredWorkGroupSizeMultiple )
//{
//  // CL_DEVICE_WARP_SIZE_NV = 0x4003.
//
//  return getDeviceInfo( 0x4003,
//                        sizeof( size_t ),
//                        &preferredWorkGroupSizeMultiple );
//}


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


OCLProcessor::OCLProcessor( int id ) :
   CachedAccelerator<OCLDevice>( id, &OCLDev ),
   _oclAdapter(),
   _cache( _oclAdapter ),
   _dma ( *this ) { }


//TODO: Configure cache awareness
void OCLProcessor::initialize()
{
   // Initialize the adapter, it will talk with the OpenCL device.
   _oclAdapter.initialize( OCLConfig::getFreeDevice() );

   // Initialize the caching subsystem.
   _cache.initialize();

   // Register this device as cache-aware.
   configureCache( _cache.getSize(), OCLConfig::getCachePolicy());
}

WD &OCLProcessor::getWorkerWD() const
{
    printf("pillando worker\n");
    OpenCLDD * dd = NEW OpenCLDD((OpenCLDD::work_fct)Scheduler::workerLoop);
    WD *wd = NEW WD(dd);
    printf("retornando worker %p\n",wd);
    return *wd;
}


WD & OCLProcessor::getMasterWD() const {    
   fatal("Attempting to create a OpenCL master thread");
}

BaseThread &OCLProcessor::createThread( WorkDescriptor &wd )
{

   ensure( wd.canRunIn( SMP ), "Incompatible worker thread" );

   OCLThread &thr = *NEW OCLThread( wd, this );

   return thr;
}

void OCLProcessor::setKernelBufferArg(void* oclKernel, int argNum, void* pointer)
{
    WD* wd=myThread->getCurrentWD();
    cl_mem buffer=_cache.toMemoryObjSS( pointer );
    //If buffer is null, this was not written, but it's allocated (output)
    //Search for it's size and assign it to an address
    if (buffer==NULL){
        CopyData *copy_array = wd->getCopies();
        unsigned int i = 0;
        for (i = 0; i < wd->getNumCopies(); ++i) {
            CopyData cpd = copy_array[i];
            //In pointers we have pointer size, not buffer size
            //This way we use the same buffer size
            void * ptr = (void*) cpd.getAddress();
            if (ptr==pointer){                
                buffer=_cache.toMemoryObjSizeSS(cpd.getSize(), pointer );
                break;;
            }
        }
    }
    //Set buffer as arg
    p_clSetKernelArg( (cl_kernel) oclKernel, argNum, sizeof(cl_mem), buffer );
}

void OCLProcessor::execKernel(void* oclKernel, 
                        int workDim, 
                        size_t* ndrOffset, 
                        size_t* ndrLocalSize, 
                        size_t* ndrGlobalSize){
     _oclAdapter.execKernel(oclKernel,
                            workDim,
                            ndrOffset,
                            ndrLocalSize,
                            ndrGlobalSize);
}