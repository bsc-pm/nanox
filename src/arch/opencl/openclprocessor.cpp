
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
  
  //Invalid context means it was already released by another thread
  if( errCode != CL_SUCCESS && errCode != CL_INVALID_CONTEXT){
     fatal0( "Unable to release the context" );
  }
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
   
   //Save OpenCL device type
   //cl_device_type devType;
   //clGetDeviceInfo( _dev, CL_DEVICE_TYPE, sizeof( cl_device_type ),&devType, NULL );
   //_preallocateWholeMemory=devType==CL_DEVICE_TYPE_CPU;

   // Create the context.
   _ctx = nanos::ext::OpenCLConfig::getContextDevice(_dev);   

   // Get a command queue.
   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_CREATE_COMMAND_QUEUE_EVENT );
   _queue = clCreateCommandQueue( _ctx, _dev, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE , &errCode );
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


void* OpenCLAdapter::allocSharedMemBuffer( size_t size )
{
   cl_int err;

   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_ALLOC_EVENT );
   cl_mem buff=clCreateBuffer(_ctx,  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,  size, NULL, &err);
   if (err!=CL_SUCCESS){
       fatal0("Failed to allocate OpenCL memory (nanos_malloc_opencl)");
   }
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;   
   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_MAP_BUFFER_SYNC_EVENT );
   void* addr= clEnqueueMapBuffer(_queue, buff, CL_TRUE, CL_MAP_READ  | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &err);   
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
   _bufCache[(void*)((size_t)addr+1)]=buff;
   return addr;
}

cl_int OpenCLAdapter::freeBuffer( cl_mem &buf )
{
   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_FREE_EVENT );
   cl_int ret= clReleaseMemObject( buf );   
   if (ret!=CL_SUCCESS){
       warning0("Failed to free OpenCL memory");
   }
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
   return ret;
}

void OpenCLAdapter::freeSharedMemBuffer( void* addr )
{
    freeBuffer(_bufCache.find(addr)->second);
    _bufCache.erase(_bufCache.find(addr)); 
}


void OpenCLAdapter::freeAddr( void* addr )
{
    freeBuffer(_bufCache.find(addr)->second);
    _bufCache.erase(_bufCache.find(addr)); 
}


cl_mem OpenCLAdapter::getBuffer( cl_mem parentBuf,
                               size_t offset,
                               size_t size)
{
   size_t baseAddress=0;
   //Cache buffers so we dont create a subBuffer everytime
   if (_bufCache.count((void*) offset)!=0){
       return _bufCache[(void*)offset];
   } 
   //If shared memory and not in the buffer cache, it must be a subbuffer
   if (OpenCLProcessor::getSharedMemAllocator().isSharedMem( (void*) offset, size)) {
       //Search the baseAddress (addres when it was allocated)
       baseAddress=(size_t )OpenCLProcessor::getSharedMemAllocator().getBasePointer( (void*) offset, size);
       parentBuf=_bufCache[(void*)(baseAddress+1)];
//       CopyData* cd=myThread->getCurrentWD()->getCopies();
//       int cdSize=myThread->getCurrentWD()->getNumCopies();
//       bool found=false;
//       for (int i=0;i < cdSize && !found; i++){
//           if (cd[i].getBaseAddress()==(void*)offset){
//               size=cd[i].getSize();
//               found=true;
//           }
//       }
       
       //Offset is address - baseAddress
       offset=offset-baseAddress;
   }
   
   if (parentBuf!=NULL){
       cl_int errCode;
       NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_CREATE_SUBBUFFER_EVENT );
       cl_buffer_region regInfo;
       regInfo.origin=offset;
       regInfo.size=size;
       cl_mem buf = clCreateSubBuffer(parentBuf,
                CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                &regInfo, &errCode);
       _bufCache[(void*)(offset+baseAddress)]=buf;
       NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
       if (errCode != CL_SUCCESS) {        
           return NULL;
       }    

       //Buf is a pointer, so this should be safe
       return buf;
   //If im a CPU (probably any kind of shared memory device), allocate buffers on demand
   //so we don't allocate the whole CPU memory
   } else {
       cl_mem buf;
       allocBuffer(size,buf);
       _bufCache[(void*)offset]=buf;
       return buf;
   }
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
                                    CL_TRUE,
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

cl_int OpenCLAdapter::mapBuffer( cl_mem buf,
                               void *dst,
                               size_t offset,
                               size_t size )
{
   cl_int errCode, exitStatus;
   cl_event ev;

   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_MAP_BUFFER_SYNC_EVENT );
   clEnqueueMapBuffer( _queue,
                                    buf,
                                    CL_TRUE,
                                    CL_MAP_READ | CL_MAP_WRITE,
                                    offset,
                                    size,
                                    0,
                                    NULL,
                                    &ev,
                                    &errCode
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
   cl_int errCode;
   cl_event ev;

   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_MEMWRITE_SYNC_EVENT );
   errCode = clEnqueueWriteBuffer( _queue,
                                     buf,
                                     CL_FALSE,
                                     offset,
                                     size,
                                     src,
                                     0,
                                     NULL,
                                     &ev
                                   );
   _pendingEvents.push_back(ev);
    NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
//   if( errCode != CL_SUCCESS ){
//      return errCode;
//   }

//   errCode = clGetEventInfo( ev,
//                               CL_EVENT_COMMAND_EXECUTION_STATUS,
//                               sizeof(cl_int),
//                               &exitStatus,
//                               NULL
//                             );
//   
//   clReleaseEvent( ev ); 

   return errCode;
}

cl_int OpenCLAdapter::unmapBuffer( cl_mem buf,
                                void *src,
                                size_t offset,
                                size_t size )
{
   cl_event ev;

   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_UNMAP_BUFFER_SYNC_EVENT );
   clEnqueueUnmapMemObject( _queue,
                                     buf,
                                     (void*) offset,
                                     0,
                                     NULL,
                                     &ev
                                   );
   //_pendingEvents.push_back(ev);
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
//   if( errCode != CL_SUCCESS ){
//      return errCode;
//   }

//   errCode = clGetEventInfo( ev,
//                               CL_EVENT_COMMAND_EXECUTION_STATUS,
//                               sizeof(cl_int),
//                               &exitStatus,
//                               NULL
//                             );
//   
//   clReleaseEvent( ev ); 

   return CL_SUCCESS;
}

cl_int OpenCLAdapter::copyInBuffer( cl_mem buf, cl_mem remoteBuffer, size_t offset_buf, size_t offset_remotebuff, size_t size ){    
   cl_int errCode;
   cl_event ev;
   
   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_COPY_BUFFER_EVENT );
   errCode = clEnqueueCopyBuffer( _queue,
                                     remoteBuffer,
                                     buf,
                                     offset_buf,
                                     offset_remotebuff,
                                     size,
                                     0,
                                     NULL,
                                     &ev
                                   );
   
   if( errCode != CL_SUCCESS ){
      return errCode;
   }
   
   _pendingEvents.push_back(ev);
   
//   errCode = clWaitForEvents(1, &ev); 
//   if( errCode != CL_SUCCESS ){
//      return errCode;
//   }

//   errCode = clGetEventInfo( ev,
//                               CL_EVENT_COMMAND_EXECUTION_STATUS,
//                               sizeof(cl_int),
//                               &exitStatus,
//                               NULL
//                             );
//   
//   clReleaseEvent( ev );

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


void* OpenCLAdapter::createKernel( const char* kernel_name, const char* ompss_code_file,const char *compilerOpts)        
{
   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_GET_PROGRAM_EVENT );
   cl_program prog;
   nanos::ext::OpenCLProcessor *pe=( nanos::ext::OpenCLProcessor * ) getMyThreadSafe()->runningOn();
   std::vector<std::string> code_files_vect;
   uint32_t hash;
   #ifndef CL_VERSION_1_2   
      std::string code_files(ompss_code_file);
      code_files_vect.push_back(code_files);
      fatal_cond0(code_files.find(",")!=std::string::npos,"In order to compile multiple OpenCL files (separated by ,), OpenCL 1.2 or higher is required");
   #else
      //Tokenize with ',' as separator
      const char* str=ompss_code_file;
      do
      {
         const char *begin = ompss_code_file;
         while(*str != ',' && *str) str++;
         code_files_vect.push_back(std::string(begin, str));
      } while (0 != *str++);
   #endif
   //When we find the kernel, stop compiling
   bool found=false;
   ProgramCache& progCache = pe->getProgCache();  
   for(std::vector<std::string>::iterator i=code_files_vect.begin(); 
        i != code_files_vect.end() && !found; 
        ++i)
   {
      std::string code_file=*i;
      #ifndef CL_VERSION_1_2
         hash= gnuHash( code_file.c_str() );
      #else
         hash= gnuHash( kernel_name );
      #endif
      //First search kernel name hash (or filename if V_CL < 1.2), then filename
      if( progCache.count( hash ) )
      {
         prog = progCache[hash];
         found=true;
      } else {
          hash= gnuHash( code_file.c_str() );
          if( progCache.count( hash ) )
          {
            prog = progCache[hash];
            found=true;      
          } else { //Compile file and add it to the progCache map (either file or kernels)
            char* ompss_code;    
            FILE *fp;
            size_t source_size;
            fp = fopen(code_file.c_str(), "r");
            fatal_cond0(!fp, "Failed to open file when loading kernel from file " + code_file + ".\n");

            fseek(fp, 0, SEEK_END); // seek to end of file;
            size_t size = ftell(fp); // get current file pointer
            fseek(fp, 0, SEEK_SET); // seek back to beginning of file
            ompss_code = new char[size+1];
            source_size = fread( ompss_code, 1, size, fp);
            (void) source_size; // FIXME: jbueno: This line avoids a Warning since source_size is set but not used.
            fclose(fp); 
            ompss_code[size]=0;

            if (code_file.find(".cl")!=code_file.npos){
               buildProgram( ompss_code, compilerOpts, prog );
            } else {
               cl_int errCode;
               const unsigned char* tmp_code=reinterpret_cast<const unsigned char*>(ompss_code);
               prog = clCreateProgramWithBinary( _ctx, 1, &_dev, &size, &tmp_code, NULL, &errCode );   
               fatal_cond0(errCode != CL_SUCCESS,"Failed to create program with binary from file " + code_file + ".\n");
               errCode = clBuildProgram( prog, 1, &_dev, compilerOpts, NULL, NULL );
               fatal_cond0(errCode != CL_SUCCESS,"Failed to create program with binary from file " + code_file + ".\n");
            }
            delete [] ompss_code;
            #ifndef CL_VERSION_1_2
               progCache[hash] = prog;
            #else
               size_t n_kernels;
               uint32_t curr_kernel_hash;
               cl_int errProg=clGetProgramInfo(prog, CL_PROGRAM_NUM_KERNELS, sizeof(size_t),&n_kernels, NULL);
               //Sometimes even with CL 1.2 it won't work, handle in runtime too
               if (errProg==CL_SUCCESS){
                   char* kernel_ids= new char[n_kernels*MAX_KERNEL_NAME_LENGTH];
                   size_t sizeret_kernels;
                   clGetProgramInfo(prog, CL_PROGRAM_KERNEL_NAMES, n_kernels*MAX_KERNEL_NAME_LENGTH*sizeof(char),kernel_ids, &sizeret_kernels);
                   if (sizeret_kernels>=n_kernels*MAX_KERNEL_NAME_LENGTH*sizeof(char))
                       warning0("Maximum kernel name length is 100 characters, you shouldn't use longer names");            

                   //Tokenize with ',' as separator            
                   str=kernel_ids;
                   do
                   {
                      const char *begin = kernel_ids;
                      while(*str != ',' && *str) str++;
                      curr_kernel_hash=gnuHash(begin, str);
                      progCache[curr_kernel_hash]=prog;
                      if (curr_kernel_hash==hash) found=true;
                   } while (0 != *str++);
                   delete[] kernel_ids;
               } else {
                   hash= gnuHash( code_file.c_str() );
                   progCache[hash]=prog;
               }
            #endif
            }
      }
   }
   
   
   cl_kernel kern;
   cl_int errCode;
   kern = clCreateKernel( prog, kernel_name, &errCode );
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
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
      std::cerr << "Error code when executing kernel " << errCode << "\n"; 
      fatal0("Error launching OpenCL kernel");
   }

   // Wait for its termination.
   errCode = clWaitForEvents( 1, &ev );
   if( errCode != CL_SUCCESS )
   {
      // Clean up environment.
      clReleaseEvent( ev );
      clReleaseKernel( openclKernel );
      std::cerr << "Error code when waiting for kernel " << errCode << "\n";
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
      fatal0("Error launching OpenCL kernel");
   }

   // Free the event.
   errCode = clReleaseEvent( ev );
   if( errCode != CL_SUCCESS )
   {
      // Clean up environment.
      clReleaseEvent( ev );
      clReleaseKernel( openclKernel );
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

void  OpenCLAdapter::waitForEvents(){
    cl_int errCode,exitStatus;
    std::vector<cl_event>::iterator iter;
    for (iter=_pendingEvents.begin() ; iter!=_pendingEvents.end() ; ++iter){
        cl_event ev;
        ev=*iter;
        errCode = clWaitForEvents(1, &ev); 
        if( errCode != CL_SUCCESS ){
            fatal0("Error waiting for events");
        }

        errCode = clGetEventInfo( ev,
                                    CL_EVENT_COMMAND_EXECUTION_STATUS,
                                    sizeof(cl_int),
                                    &exitStatus,
                                    NULL
                                  );
        if( errCode != CL_SUCCESS ){
            fatal0("Error waiting for events");
        }
        
        clReleaseEvent( ev );
        NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
    }
    _pendingEvents.clear();
}

//
// OpenCLProcessor implementation.
//

SharedMemAllocator OpenCLProcessor::_shmemAllocator;

OpenCLProcessor::OpenCLProcessor( int id, int devId, int uid ) :
   CachedAccelerator<OpenCLDevice>( id, &OpenCLDev, uid ),
   _openclAdapter(),
   _cache( _openclAdapter ),
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
   waitForEvents();
   message("OpenCL dev" << _devId << " TRANSFER STATISTICS");
   message("    Total input transfers: " << bytesToHumanReadable( _cache._bytesIn.value() ) );
   message("    Total output transfers: " << bytesToHumanReadable( _cache._bytesOut.value() ) );
   message("    Total dev2dev(in) transfers: " << bytesToHumanReadable( _cache._bytesDevice.value() ) );
}
 

void OpenCLProcessor::cleanUp()
{
   printStats();
}

void* OpenCLProcessor::allocateSharedMemory( size_t size ){    
    return _openclAdapter.allocSharedMemBuffer(size);
}

void OpenCLProcessor::freeSharedMemory( void* addr ){    
    _openclAdapter.freeSharedMemBuffer((void*)((size_t)addr+1));
}