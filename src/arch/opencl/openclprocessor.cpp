/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#include "openclprocessor.hpp"
#include "openclthread.hpp"
#include "smpprocessor.hpp"
#include "os.hpp"
#include "openclevent.hpp"
#include <iostream>
#include <algorithm>

#if defined(__GNUC__) && __GNUC__ > 4
// This is not understood by icpc
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace nanos;
using namespace nanos::ext;

//
// OpenCLAdapter implementation.
//

OpenCLAdapter::~OpenCLAdapter()
{
  cl_int errCode;

  
  for (int i=0; i<nanos::ext::OpenCLConfig::getPrefetchNum(); ++i) {
     errCode = clReleaseCommandQueue( _queues[i] ); 
     if( errCode != CL_SUCCESS )
        warning0( "Unable to release the command queue" );
  }
  delete[] _queues;
  
  errCode = clReleaseCommandQueue( _copyInQueue );
  if( errCode != CL_SUCCESS )
     warning0( "Unable to release the in transfers command queue" );
  errCode = clReleaseCommandQueue( _copyOutQueue );
  if( errCode != CL_SUCCESS )
     warning0( "Unable to release the out transfers command queue" );
  errCode = clReleaseCommandQueue( _profilingQueue );
  if( errCode != CL_SUCCESS )
     warning0( "Unable to release the profiling command queue" );
  
  // Release the track memory of profiling executions
  for ( std::map<std::string, DimsBest>::iterator kernelIt = _bestExec.begin(); kernelIt != _bestExec.end(); kernelIt++ )
  {
     // Clean best executions
     DimsBest &dimsBest = kernelIt->second;
     for (DimsBest::iterator dimIt = dimsBest.begin(); dimIt != dimsBest.end(); dimIt++ )
     {
        delete dimIt->second;
     }
     dimsBest.clear();

     // Clean statistics
     DimsExecutions &dimsExecutions = _nExecutions[kernelIt->first];
     dimsExecutions.clear();
  }
  _bestExec.clear();
  _nExecutions.clear();

  if ( OpenCLConfig::isEnableProfiling() ) {
    // Closing the database and deallocating the object
    delete _openCLProfilerDbManager;
  }

  for( ProgramCache::iterator i = _progCache.begin(),
                              e = _progCache.end();
                              i != e;
                              ++i )
    clReleaseProgram( i->second );

  errCode = clReleaseContext( _ctx );
  
  //Invalid context means it was already released by another thread
  if( errCode != CL_SUCCESS && errCode != CL_INVALID_CONTEXT){
     warning0( "Unable to release the context" );
  }
}

void OpenCLAdapter::initialize(cl_device_id dev)
{
   cl_int errCode;

   _dev = dev;
   
   //Save OpenCL device type
   //cl_device_type devType;
   //clGetDeviceInfo( _dev, CL_DEVICE_TYPE, sizeof( cl_device_type ),&devType, NULL );
   //_useHostPtrs= (devType==CL_DEVICE_TYPE_CPU);
   _useHostPtrs= false;
   cl_int align;
   clGetDeviceInfo(_dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(align), &align, NULL);
   verbose("CL device align: " << align);
   
   _useHostPtrs=_useHostPtrs || nanos::ext::OpenCLConfig::getForceShMem();

   // Create the context.
   _ctx = nanos::ext::OpenCLConfig::getContextDevice(_dev);   
   
   //Using host Pointers is not compatible with preallocating whole memory mode
   if (_useHostPtrs) _preallocateWholeMemory=false;

   // Get a command queue.
   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_CREATE_COMMAND_QUEUE_EVENT );
   _queues = NEW cl_command_queue[nanos::ext::OpenCLConfig::getPrefetchNum()];
   for (int i=0; i<nanos::ext::OpenCLConfig::getPrefetchNum(); ++i) {
      _queues[i] = clCreateCommandQueue( _ctx, _dev, 0 , &errCode );
      if( errCode != CL_SUCCESS )
         fatal0( "Cannot create a command queue" );
   }
   _currQueue=0;
   _copyInQueue = clCreateCommandQueue( _ctx, _dev, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE , &errCode );
   if( errCode != CL_SUCCESS )
     fatal0( "Cannot create a command queue for in transfers" );
   _copyOutQueue = clCreateCommandQueue( _ctx, _dev, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE , &errCode );
   if( errCode != CL_SUCCESS )
     fatal0( "Cannot create a command queue" );
   _profilingQueue = clCreateCommandQueue( _ctx, _dev, CL_QUEUE_PROFILING_ENABLE , &errCode );
   if( errCode != CL_SUCCESS )
     fatal0( "Cannot create a command queue for profiling" );

   std::string deviceVendor = getDeviceVendor();
   setSynchronization(deviceVendor);

   if ( OpenCLConfig::isEnableProfiling() ) {
     // Allocating the object and opening the database
     _openCLProfilerDbManager = new OpenCLProfilerDbManager();
   }

   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
}

void OpenCLAdapter::setSynchronization( std::string &vendor )
{
	std::transform(vendor.begin(), vendor.end(), vendor.begin(), toupper);
	if ( vendor.find("INTEL") != std::string::npos ) {
		_synchronize = true;
	} else 	if ( vendor.find("ARM") != std::string::npos ) {
		_synchronize = true;
	}
}

cl_int OpenCLAdapter::allocBuffer( size_t size, void* host_ptr, cl_mem &buf )
{
   cl_int errCode;

   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_ALLOC_EVENT );
   if (_useHostPtrs) {
      buf = clCreateBuffer( _ctx, CL_MEM_USE_HOST_PTR, size, host_ptr, &errCode );
   } else {
      buf = clCreateBuffer( _ctx, CL_MEM_READ_WRITE, size, NULL, &errCode );       
   }
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
   void* addr= clEnqueueMapBuffer(_copyOutQueue, buff, CL_TRUE, CL_MAP_READ  | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &err);   
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
   _bufCache[std::make_pair((uint64_t)addr,size)]=buff;
   _sizeCache[(uint64_t)addr]=size;
   return addr;
}

cl_int OpenCLAdapter::freeBuffer( cl_mem &buf )
{
   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_FREE_EVENT );
   cl_int ret= clReleaseMemObject( buf );   
   if (ret!=CL_SUCCESS){
       //warning0("Failed to free OpenCL memory");
       //std::cerr << "Error number: " << ret << "," << buf << "\n";
   }
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
   return ret;
}

void OpenCLAdapter::freeSharedMemBuffer( void* addr )
{
    //Right now is the same
    freeAddr(addr);
}


void OpenCLAdapter::freeAddr(void* addrin )
{
   uint64_t addr=(uint64_t)addrin;
   size_t old_size=_sizeCache[(uint64_t)addr];    
  
   BufferCache::iterator iter=_bufCache.begin();
   //Search OCL cache looking for the buffer and also sub-buffers
   while (iter != _bufCache.end())
   {
       uint64_t addr_cache=iter->first.first;
       size_t size_cache=iter->first.second;
       
       if (addr <= addr_cache && addr+old_size >= addr_cache+size_cache ) {
           freeBuffer(iter->second);
           _bufCache.erase(iter++);
           _sizeCache.erase(addr_cache);
       } else {
           ++iter;
       }
   }
}

size_t OpenCLAdapter::getSizeFromCache(uint64_t addr){
    return _sizeCache[addr];
}

cl_mem OpenCLAdapter::getBuffer(SimpleAllocator& allocator, cl_mem parentBuf,
                               uint64_t devAddr,
                               size_t size)
{
   std::pair<uint64_t,size_t> cacheKey= std::make_pair(devAddr,size);
   //If this exact buffer is in cache, return
   if (_bufCache.count(cacheKey)!=0){
       return _bufCache[cacheKey];
   } 
   
   bool isSharedMem=OpenCLProcessor::getSharedMemAllocator().isSharedMem( (void*) devAddr, size);
   
   //If there is a buffer which covers this buffer (same base address but bigger), return it
   uint64_t baseAddress;
   if (isSharedMem) {       
      baseAddress=reinterpret_cast<uint64_t>(OpenCLProcessor::getSharedMemAllocator().getBasePointer( (void*) devAddr, size)) ;
   } else {
      //If there is a buffer which covers this buffer (same base address but bigger), return it
      baseAddress=allocator.getBasePointer(devAddr, size);     
   }
   
   if (baseAddress==devAddr && size<=_sizeCache[baseAddress]){
       cacheKey= std::make_pair(devAddr,_sizeCache[baseAddress]);
       return _bufCache[cacheKey];
   }
   
   //Free the current buffer if it's the same pointer and different size
   //Except when it's not a subbuffer (main cache controls these)
   //The second part of this if shouldn't be needed (two previous ifs cover the 
   //case of new_size < buffSize and new_size == buffSize, so nanox cache should have reallocated this)
   if (_sizeCache.count(devAddr)!=0 && baseAddress!=devAddr && baseAddress!= 0 )  {
       freeAddr((void*)devAddr);
   }

   
   size_t old_size=_sizeCache[baseAddress];       
   parentBuf=_bufCache[std::make_pair(baseAddress,old_size)];
   //Get the offset from the baseaddress (-1 in case shared mem)
   devAddr=devAddr-baseAddress;
   
   //Now create the subbuffer (either from sharedMemory, mainBuffer when in prealloc mode, or from its "baseBuffer" in normal mode)
   if (parentBuf!=NULL){
       cl_int errCode;
       NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_CREATE_SUBBUFFER_EVENT );
       cl_buffer_region regInfo;
       regInfo.origin=devAddr;
       regInfo.size=size;
       cl_mem buf = clCreateSubBuffer(parentBuf,
                CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                &regInfo, &errCode);

	   if (errCode != CL_SUCCESS) {
		  if (errCode == CL_MISALIGNED_SUB_BUFFER_OFFSET) {
			 std::cerr << "Error trying to create a subbuffer whose offset "
					   <<  "is not properly aligned." << std::endl;

			 // The specification says that it has to be aligned to
			 // CL_DEVICE_MEM_BASE_ADDR_ALIGN. However, sometimes work with
			 // other values (depending on the vendor)
		  }
		  fatal0("Error creating a subbuffer");
	   }

       _bufCache[std::make_pair(devAddr+baseAddress,size)]=buf;
       _sizeCache[devAddr+baseAddress]=size;
       NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
       if (errCode != CL_SUCCESS) {      
           return NULL;
       }    

       //Buf is a pointer, so this should be safe
       return buf;
   } else {       
       fatal0("Error in OpenCL cache, tried to get a buffer which was not allocated before");
   }
}


cl_mem OpenCLAdapter::createBuffer(cl_mem parentBuf,
                               uint64_t devAddr,
                               size_t size,
                               void* hostPtr)
{
   //Preallote whole memory mode
   if (parentBuf!=NULL){
       cl_int errCode;
       NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_CREATE_SUBBUFFER_EVENT );
       cl_buffer_region regInfo;
       regInfo.origin=devAddr-ALLOCATOR_START_ADDR;
       regInfo.size=size;
       cl_mem buf = clCreateSubBuffer(parentBuf,
                CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                &regInfo, &errCode);
       _bufCache[std::make_pair(devAddr,size)]=buf;
       _sizeCache[devAddr]=size;      
       NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
       if (errCode != CL_SUCCESS) {  
           warning("Error when creating subBuffer from preallocated memory " << errCode);
           return NULL;
       }    

       //Buf is a pointer, so this should be safe
       return buf;
   //If not in preallocation mode
   } else {
       cl_mem buf;
       allocBuffer(size, hostPtr, buf);
       _bufCache[std::make_pair(devAddr,size)]=buf;
       _sizeCache[devAddr]=size;
       return buf;
   }
}

cl_int OpenCLAdapter::readBuffer(cl_mem buf,
        void *dst,
        size_t offset,
        size_t size,
        Atomic<size_t>* globalSizeCounter,
        cl_event& ev) {
    cl_int ret;
    if (_useHostPtrs || OpenCLProcessor::getSharedMemAllocator().isSharedMem( (void*) ((uint64_t)dst+offset), size)) {
        if (_useHostPtrs) *globalSizeCounter += size;
        ret = mapBuffer(buf, dst, offset, size, ev);
    } else {

        NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT(ext::NANOS_OPENCL_MEMREAD_SYNC_EVENT);
        ret = clEnqueueReadBuffer(_copyOutQueue,
                buf,
                CL_FALSE,
                offset,
                size,
                dst,
                0,
                NULL,
                &ev
                );

        if ( _synchronize )
     	   clWaitForEvents(1, &ev);

        *globalSizeCounter+=size;
        NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;      
    }
    return ret;
}

cl_int OpenCLAdapter::mapBuffer( cl_mem buf,
                               void *dst,
                               size_t offset,
                               size_t size,
                               cl_event& ev )
{
   cl_int errCode;
   _unmapedCache.erase(buf);

   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_MAP_BUFFER_SYNC_EVENT );
   clEnqueueMapBuffer( _copyOutQueue,
                                    buf,
                                    CL_FALSE,
                                    CL_MAP_READ | CL_MAP_WRITE,
                                    offset,
                                    size,
                                    0,
                                    NULL,
                                    &ev,
                                    &errCode
                                  );

   if ( _synchronize )
      clWaitForEvents(1, &ev);

   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
      
   return errCode;
}

cl_int OpenCLAdapter::writeBuffer( cl_mem buf,
                                void *src,
                                size_t offset,
                                size_t size,
                                Atomic<size_t>* globalSizeCounter,
                                cl_event& ev)
{
   cl_int ret;
   if (_useHostPtrs || OpenCLProcessor::getSharedMemAllocator().isSharedMem( (void*) ((uint64_t)src+offset), size)) {
       if (_useHostPtrs) *globalSizeCounter += size;
       ret=unmapBuffer(buf,src,offset,size,ev);
   } else {

       NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_MEMWRITE_SYNC_EVENT );
       ret = clEnqueueWriteBuffer( _copyInQueue,
                                          buf,
                                          CL_FALSE,
                                          offset,
                                          size,
                                          src,
                                          0,
                                          NULL,
                                          &ev
                                        );

       if ( _synchronize )
    	   clWaitForEvents(1, &ev);

        *globalSizeCounter += size;
        NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
   }
   return ret;
}

cl_int OpenCLAdapter::unmapBuffer(cl_mem buf,
        void *src,
        size_t offset,
        size_t size,
        cl_event& ev) {
    
    if (_unmapedCache.count(buf) != 0) {
        ev = NULL;
        return CL_SUCCESS;
    }
    cl_int errCode;

    NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT(ext::NANOS_OPENCL_UNMAP_BUFFER_SYNC_EVENT);

    errCode = clEnqueueUnmapMemObject(_copyInQueue,
            buf,
            src,
            0,
            NULL,
            &ev
            );

    if ( _synchronize )
       clWaitForEvents(1, &ev);

    //This is a dirty trick to fake OpenCL driver which only accepts unmaps of previously mapped values
    //mapping something which is on the CPU should not do anything bad.
    //basically we map the buffer and then unmap it.
    if (errCode == CL_INVALID_VALUE) {
        mapBuffer(buf, src, offset, size, ev);
        errCode = clEnqueueUnmapMemObject(_copyInQueue,
                buf,
                src,
                0,
                NULL,
                &ev
                );
         if (errCode != CL_SUCCESS) {
             fatal0("Errror unmapping buffer");
         }
    }
    
    NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;

    _unmapedCache.insert(std::make_pair<cl_mem, int>(buf, 0));

    return CL_SUCCESS;
}

cl_int OpenCLAdapter::copyInBuffer( cl_mem buf, 
                                    cl_mem remoteBuffer, 
                                    size_t offset_buf, 
                                    size_t offset_remotebuff, 
                                    size_t size,
                                    cl_event& ev ){    
   cl_int errCode;
   
   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_COPY_BUFFER_EVENT );
   errCode = clEnqueueCopyBuffer( _copyInQueue,
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
   
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
   
   
   return errCode;
   
}

cl_int OpenCLAdapter::buildProgram( const char *src,
                                 const char *compilerOpts,
                                 cl_program &prog,
                                 const std::string& filename )
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
#if 0 /* master version */
   else if( OpenCLConfig::isSaveKernelEnabled() )
   {
      std::string binaryFilename = filename.substr( 0, filename.find(".cl") ) + ".bin";
      
      size_t sizeRet;
      errCode = clGetProgramInfo( prog, CL_PROGRAM_BINARY_SIZES, 0, NULL, &sizeRet );
      if ( errCode != CL_SUCCESS ) return errCode;

      size_t binarySize;
      errCode = clGetProgramInfo( prog, CL_PROGRAM_BINARY_SIZES, sizeRet, &binarySize, NULL );
      if ( errCode != CL_SUCCESS ) return errCode;

      unsigned char* binary = (unsigned char *) alloca (sizeof (unsigned char) * binarySize);
      fatal_cond( binary == NULL, "Call to alloca failed" );
      errCode = clGetProgramInfo( prog, CL_PROGRAM_BINARIES, sizeof (unsigned char *), &binary, NULL );
      if ( errCode != CL_SUCCESS ) return errCode;

       FILE *fp = fopen (binaryFilename.c_str(), "w");
       if ( fp == NULL ) fatal( "Error opening OpenCL kernel binary file for writing" );

       fatal_cond( fwrite( binary, 1, binarySize, fp ) != binarySize, "Error writing OpenCL kernel binary file" );

       fclose( fp );
   }
#endif
   return errCode;
}

cl_int OpenCLAdapter::destroyProgram( cl_program &prog )
{
  return clReleaseProgram( prog );
}

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
               buildProgram( ompss_code, compilerOpts, prog, code_file );
            } else {
               cl_int errCode;
               const unsigned char* tmp_code=reinterpret_cast<const unsigned char*>(ompss_code);
               prog = clCreateProgramWithBinary( _ctx, 1, &_dev, &size, &tmp_code, NULL, &errCode );   
               fatal_cond0(errCode != CL_SUCCESS,"Failed to create program with binary from file " + code_file + ". If this file is not a binary file"
                       " please use \".cl\" extension\n");
               errCode = clBuildProgram( prog, 1, &_dev, compilerOpts, NULL, NULL );
               fatal_cond0(errCode != CL_SUCCESS,"Failed to create program with binary from file " + code_file + ". If this file is not a binary file"
                       " please use \".cl\" extension\n");;
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

                   //Tokenize with ';' as separator     
                   str=kernel_ids;
                   do
                   {
                      const char *begin = str;
                      while( *str != ';' && *str) str++;
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

static void processOpenCLError(cl_int errCode) {
   std::cerr << "Error code when executing kernel " << errCode << "\n";
   switch (errCode) {
      case CL_OUT_OF_RESOURCES: // -5
         {
            std::cerr
               << "HINT: Out of resources, make sure that ndrange local size "
               << "fits in your device or that your kernel is not reading/writing outside of the buffer\n";
            break;
         }
      case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: // -14
         {
            std::cerr
               << "HINT: Check if your input or output data sizes are correctly specified/accessed\n";
            break;
         }
      case CL_INVALID_KERNEL_ARGS: // -52
         {
            std::cerr
               << "HINT: Check if the OpenCL kernel declaration in the "
               << "header/interface file and the definition in .cl have the same parameters\n";
            break;
         }
      default:
         {
            // We don't have any hint for this error
            break;
         }
   }
}

void OpenCLAdapter::execKernel(void* oclKernel, 
                        int workDim, 
                        size_t* ndrOffset, 
                        size_t* ndrLocalSize, 
                        size_t* ndrGlobalSize)
{
   cl_kernel openclKernel=(cl_kernel) oclKernel;
   OpenCLThread* thread= (OpenCLThread*) myThread;
   cl_int errCode;
   
   OpenCLEvent* oclEvent= (OpenCLEvent*) thread->getCurrKernelEvent();
   //Set the kernel so the OCL event can free it when it finishes
   oclEvent->setCLKernel(oclKernel);
   WD* wd=myThread->getCurrentWD();
   int currQueue=_currQueue;
   OpenCLDD& curdd=static_cast<OpenCLDD&>(wd->getActiveDevice());
   
   if ( curdd.getOpenCLStreamIdx() == -1 ) {
      curdd.setOpenclStreamIdx( currQueue );
   } else {
      currQueue = curdd.getOpenCLStreamIdx();
   }

   debug0( "[opencl] global size: " + toString( *ndrGlobalSize ) + ", local size: " + toString( *ndrLocalSize ) );
   // Exec it.
   errCode = clEnqueueNDRangeKernel( _queues[currQueue],
                                       openclKernel,
                                       workDim,
                                       ndrOffset,
                                       ndrGlobalSize,
                                       ndrLocalSize,
                                       0,
                                       NULL,
                                       &oclEvent->getCLEvent()
                                     );

   if ( _synchronize )
	   clWaitForEvents(1, &oclEvent->getCLEvent());

   if ( currQueue == _currQueue ) {
      _currQueue = ( _currQueue + 1 )  % nanos::ext::OpenCLConfig::getPrefetchNum();
   }
                                     
   if( errCode != CL_SUCCESS )
   {
      // Don't worry about exit code, we are cleaning an error.
      clReleaseKernel( openclKernel );
      processOpenCLError(errCode);
      oclEvent->setCLKernel(NULL);
      fatal0kernelNameErr(oclKernel,"Error launching OpenCL kernel",errCode);
   }
}

void OpenCLAdapter::profileKernel(void* oclKernel,
         int workDim,
         int range_size,
         size_t* ndrOffset,
         size_t* ndrLocalSize,
         size_t* ndrGlobalSize)
{
   size_t local_work_size[3];
   const double cost = 0;

   const std::string kernelName = getKernelName(oclKernel);

#ifdef NANOS_DEBUG_ENABLED
   debug( "[opencl] Profiling kernel: " + toString( oclKernel ) + ". Profiling configuration:" );
   for ( int i=0; i<workDim; i++ )
   {
      debug( "[opencl] Dimension: " + toString( i ) );
      for ( int j=0; j<range_size; j++ )
      {
         debug( "[opencl]    [" + toString( j ) + "]Global size: " + toString( ndrGlobalSize[i*range_size+j] ) );
         debug( "[opencl]    [" + toString( j ) + "]Local size:  " + toString( ndrLocalSize[i*range_size+j] ) );
      }
   }
#endif

   Dims dims(workDim, ndrGlobalSize[0], ndrGlobalSize[1], ndrGlobalSize[2], cost);

   // Check whether Nanos already have the best configuration from this execution
   Execution *bestExecution = NULL;
   if ( _bestExec.count(kernelName) > 0 ) {
      // We have at least one execution for this kernel
      DimsBest &dimsBest = _bestExec[kernelName];

      if ( dimsBest.count(dims) > 0 ) {
         // We already have the best execution with these dimensions
         bestExecution = dimsBest[dims];
      }
   }

   // Nanos does not have the best configuration from this execution
   // Access to the database and search for an optimal configuration
   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_PROFILE_DB_ACCESS );
   if ( bestExecution == NULL ) {
      if ( getOpenClProfilerDbManager() == NULL )
         fatal0("OpenCL Profiler flag was not provided")
      bestExecution = getOpenClProfilerDbManager()->getKernelConfig(dims, kernelName);
      if ( bestExecution->getNdims() < 9 ) {
         Execution *bestExecutionTmp = new  Execution(*bestExecution);
         updateProfiling(kernelName, bestExecutionTmp, dims);
      }
   }
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;

   if ( bestExecution != NULL && bestExecution->getNdims() < 9 ) {
     // Run with the best execution
     local_work_size[0] = bestExecution->getLocalX();
     local_work_size[1] = bestExecution->getLocalY();
     local_work_size[2] = bestExecution->getLocalZ();
     // Do not profile
     execKernel(oclKernel, workDim, ndrOffset, local_work_size, ndrGlobalSize);
   }
   else {
      // Start to profile the kernel
      if ( true /* whatever */ )
         manualProfileKernelStep(oclKernel, kernelName, workDim, range_size, cost, dims, ndrOffset, ndrLocalSize, ndrGlobalSize);
      else
         smartProfileKernel(oclKernel, kernelName, workDim, range_size, cost, dims, ndrOffset, ndrGlobalSize);

     // New configuration found. Save it
     NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_PROFILE_DB_ACCESS );
     DimsBest dimsBest = _bestExec[kernelName];
     assert(dimsBest[dims] != NULL);
     Execution execution(*dimsBest[dims]);
     getOpenClProfilerDbManager()->setKernelConfig(dims, execution, kernelName);
     NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
   }
}

void OpenCLAdapter::manualProfileKernel( void* oclKernel,
                                         std::string kernelName,
                                         int workDim,
                                         int range_size,
                                         const double cost,
                                         Dims &dims,
                                         size_t* ndrOffset,
                                         size_t* ndrLocalSize,
                                         size_t* ndrGlobalSize)
{
   size_t local_work_size[3], global_work_size[3];

   // Limit the iterations based on number of dimensions
   const int zLimit = workDim == 3 ? range_size : 1;
   const int yLimit = workDim == 2 ? range_size : 1;

   for ( int z=0; z<zLimit; z++ )
   {
      local_work_size[2] = ndrLocalSize[2*range_size+z];
      global_work_size[2] = ndrGlobalSize[2*range_size+z];
      for ( int y=0; y<yLimit; y++ )
      {
         local_work_size[1] = ndrLocalSize[range_size+y];
         global_work_size[1] = ndrGlobalSize[range_size+y];
         for ( int x=0; x<range_size; x++ )
         {
            local_work_size[0] = ndrLocalSize[x];
            global_work_size[0] = ndrGlobalSize[x];
            updateProfiling(kernelName, singleExecKernel(oclKernel, workDim, ndrOffset, local_work_size, global_work_size), dims);
         }
      }
   }
}

void OpenCLAdapter::manualProfileKernelStep( void* oclKernel,
                                         std::string kernelName,
                                         int workDim,
                                         int range_size,
                                         const double cost,
                                         Dims &dims,
                                         size_t* ndrOffset,
                                         size_t* ndrLocalSize,
                                         size_t* ndrGlobalSize)
{
   size_t local_work_size[3], global_work_size[3];

   // Limit the iterations based on number of dimensions
   const int zLimit = workDim == 3 ? range_size : 1;
   const int yLimit = workDim == 2 ? range_size : 1;

   for ( int z=0; z<zLimit; z++ )
   {
      local_work_size[2] = ndrLocalSize[2*range_size+z];
      global_work_size[2] = ndrGlobalSize[2*range_size+z];
      for ( int y=0; y<yLimit; y++ )
      {
         local_work_size[1] = ndrLocalSize[range_size+y];
         global_work_size[1] = ndrGlobalSize[range_size+y];
         for ( int x=0; x<range_size; x++ )
         {
            local_work_size[0] = ndrLocalSize[x];
            global_work_size[0] = ndrGlobalSize[x];
            updateProfiling(kernelName, singleExecKernel(oclKernel, workDim, ndrOffset, local_work_size, global_work_size), dims);
         }
      }
   }
}

void OpenCLAdapter::smartProfileKernel(void* oclKernel,
                                       std::string kernelName,
                                       int workDim,
                                       int range_size,
                                       const double cost,
                                       Dims &dims,
                                       size_t* ndrOffset,
                                       size_t* ndrGlobalSize)
{
   size_t local_work_size[3], global_work_size[3];
   cl_kernel kernel = (cl_kernel) oclKernel;

   // Device work-groups bounds
   const short safeWorkGroupMultiple = 2;
   const short safeMaxWorkGroup = 64;
   const short limitWorkGroupMultiple = 128;
   const short limitMaxWorkGroup = 1024;
   if ( _maxWorkGroup == 0 )
      getMaxWorkGroup(kernel);
   if ( _workGroupMultiple == 0 )
      getWorkGroupMultiple(kernel);

   // Safe check for Intel drivers
   const size_t maxGlobal = std::max(ndrGlobalSize[0], std::max(ndrGlobalSize[1], ndrGlobalSize[2]));
   if ( _workGroupMultiple > limitWorkGroupMultiple || _workGroupMultiple < 1 || _workGroupMultiple > maxGlobal ) {
      _workGroupMultiple = safeWorkGroupMultiple;
      _workGroupMultiple = _workGroupMultiple > maxGlobal ? maxGlobal : _workGroupMultiple;
      std::stringstream msg;
      msg << " OpenCL Profiling: Applying safe work-group multiple value to ";
      msg << _workGroupMultiple;
      warning( msg.str() )
   }
   if ( _maxWorkGroup > limitMaxWorkGroup || _maxWorkGroup < 1) {
      // Device check
      _maxWorkGroup = _workGroupMultiple > safeMaxWorkGroup ? safeMaxWorkGroup : _workGroupMultiple ;
      std::stringstream msg;
      msg << " OpenCL Profiling: max. work-group value to ";
      msg << _maxWorkGroup;
      warning( msg.str() )
   }

   // Limit the iterations based on number of dimensions
   unsigned int multiplePreferred;
   unsigned int zLimit = _workGroupMultiple;
   unsigned int yLimit = _workGroupMultiple;
   switch ( workDim )
   {
      case 1:
      {
         multiplePreferred = _workGroupMultiple;
         break;
      }
      case 2:
      {
         multiplePreferred = _workGroupMultiple/2;
         yLimit = _maxWorkGroup;
         break;
      }
      case 3:
      {
         multiplePreferred = _workGroupMultiple/4;
         zLimit = _maxWorkGroup;
         yLimit = _maxWorkGroup;
         break;
      }
      default:
         multiplePreferred = _workGroupMultiple;
   }

   for ( int z=1; _workGroupMultiple*z<=zLimit; z++ )
   {
      local_work_size[2] = multiplePreferred*z;
      global_work_size[2] = ndrGlobalSize[2*range_size+z];
      for ( int y=1; _workGroupMultiple*z*y<=yLimit; y++ )
      {
         local_work_size[1] = multiplePreferred*y;
         global_work_size[1] = ndrGlobalSize[range_size+y];
         for ( int x=1; _workGroupMultiple*z*y*x<=_maxWorkGroup; x++ )
         {
            local_work_size[0] = multiplePreferred*x;
            global_work_size[0] = ndrGlobalSize[x];
            updateProfiling(kernelName, singleExecKernel(oclKernel, workDim, ndrOffset, local_work_size, global_work_size), dims);
         }
      }
   }
}

Execution* OpenCLAdapter::singleExecKernel(void* oclKernel,
         int workDim,
         size_t* ndrOffset,
         size_t* ndrLocalSize,
         size_t* ndrGlobalSize)
{
   cl_kernel openclKernel=(cl_kernel) oclKernel;
   cl_int errCode;
   cl_event event;

   debug0( "[opencl] global size: " + toString( *ndrGlobalSize ) + ", local size: " + toString( *ndrLocalSize ) );
   // Exec it.

   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_PROFILE_KERNEL );

   errCode = clEnqueueNDRangeKernel( _profilingQueue,
            openclKernel,
            workDim,
            ndrOffset,
            ndrGlobalSize,
            ndrLocalSize,
            0,
            NULL,
            &event
   );

   if( errCode != CL_SUCCESS )
   {
      // Don't worry about exit code, we are cleaning an error.
      clReleaseKernel( openclKernel );
      processOpenCLError(errCode);
      fatal0kernelNameErr(oclKernel,"Error launching OpenCL kernel",errCode);
   }

   clCheckError(clWaitForEvents(1, &event), (char *) "Error when waiting for events");

   cl_ulong startTime, endTime;
   clCheckError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL), (char *)"Error reading start execution");
   clCheckError(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL), (char *)"Error reading end execution");

   unsigned int localX, localY, localZ;
   localX = ndrLocalSize[0];
   if ( workDim > 1) {
      localY = ndrLocalSize[1];
      if ( workDim > 2) {
         localZ = ndrLocalSize[2];
      }
      else {
         localZ = 0;
      }
   } else {
      localY = localZ = 0;
   }

   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;

   return new Execution(workDim,localX,localY,localZ,endTime-startTime);
}

void OpenCLAdapter::updateProfiling(std::string kernelName, Execution *execution, Dims& dims)
{
   NANOS_OPENCL_CREATE_IN_OCL_RUNTIME_EVENT( ext::NANOS_OPENCL_PROFILE_UPDATE_DATA );
   if ( _bestExec.count(kernelName) > 0 ) {
      // We have at least one execution for this kernel
      DimsBest &dimsBest = _bestExec[kernelName];
      DimsExecutions &dimsExecutions = _nExecutions[kernelName];

      if ( dimsBest.count(dims) ) {
         // We have at least one execution with these dimensions
         Execution *bestExecution = dimsBest[dims];

         // Update best execution
         if ( *execution < *bestExecution )
         {
            dimsBest[dims] = execution;
            delete bestExecution;
         }

         dimsExecutions[dims]++;
      } else {
         // We do not have any executions with these dimensions for this kernel
         dimsBest[dims] = execution;
         dimsExecutions[dims] = 1;
      }
   } else {
      // This is the first execution for this kernel
      DimsBest *dimsBest = new DimsBest;
      (*dimsBest)[dims] = execution;
      _bestExec[kernelName] = *dimsBest;

      DimsExecutions *dimsExecutions = new DimsExecutions;
      (*dimsExecutions)[dims] = 1;
      _nExecutions[kernelName] = *dimsExecutions;
   }
   NANOS_OPENCL_CLOSE_IN_OCL_RUNTIME_EVENT;
}

void OpenCLAdapter::printProfiling()
{
   if ( _bestExec.size() > 0 ) {
      std::cout.precision(3);
      std::cout << "-------------------------------------------------" << std::endl;
      std::cout << "OpenCL Performance Profile" << std::endl;
      std::cout << "-------------------------------------------------" << std::endl;
      for ( std::map<std::string, DimsBest>::const_iterator dimsBestIt = _bestExec.begin(); dimsBestIt != _bestExec.end(); dimsBestIt++ )
      {
         std::string kernelName = dimsBestIt->first;
         DimsExecutions &dimsExecutions = _nExecutions[kernelName];
         std::cout << "#################################################" << std::endl;
         std::cout << "Kernel name: " << kernelName << std::endl;
         std::cout << "#################################################" << std::endl;
         for ( DimsBest::const_iterator dim = dimsBestIt->second.begin(); dim != dimsBestIt->second.end(); dim++ )
         {
            Dims currDims = dim->first;
            Execution *bestExecution = dim->second;
            double performance = currDims.getCost()/(bestExecution->getTime()/1.e9f);

            std::cout << "................................................." << std::endl;
            std::cout << "Dimensions: ";
            switch ( currDims.getNdims() ) {
               case 1:
                  std::cout << "X=" << currDims.getGlobalX() << std::endl;
                  break;
               case 2:
                  std::cout << "X=" << currDims.getGlobalX() << ", Y=" << currDims.getGlobalY() << std::endl;
                  break;
               case 3:
                  std::cout << "X=" << currDims.getGlobalX() << ", Y=" << currDims.getGlobalY() << ", Z=" << currDims.getGlobalZ() << std::endl;
                  break;
               default:
                  throw nanos::OpenCLProfilerException(CLP_WRONG_NUMBER_OF_DIMENSIONS);
            }
            std::cout << "Best configuration found (work-group dimensions):" << std::endl;
            switch ( bestExecution->getNdims() ) {
               case 1:
                  std::cout << "X=" << bestExecution->getLocalX() << std::endl;
                  break;
               case 2:
                  std::cout << "X=" << bestExecution->getLocalX() << ", Y=" << bestExecution->getLocalY() << std::endl;
                  break;
               case 3:
                  std::cout << "X=" << bestExecution->getLocalX() << ", Y=" << bestExecution->getLocalY() << ", Z=" << bestExecution->getLocalZ() << std::endl;
                  break;
               default:
                  throw nanos::OpenCLProfilerException(CLP_WRONG_NUMBER_OF_DIMENSIONS);
            }
            std::cout << "Best execution time (in ns): " << bestExecution->getTime() << std::endl;
            std::cout << "Total configurations tested: " << dimsExecutions[currDims] << std::endl;
            if ( performance > 0 )
               std::cout << "Performance: " << performance << " Gflops" << std::endl;
            std::cout << "................................................." << std::endl;
         }
      }
      std::cout << "-------------------------------------------------" << std::endl;
   }
}

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

std::string OpenCLAdapter::getDeviceName(){ 
   char* value;
   size_t valueSize;
   clGetDeviceInfo(_dev, CL_DEVICE_NAME, 0, NULL, &valueSize);
   value = (char*) malloc(valueSize);
   clGetDeviceInfo(_dev, CL_DEVICE_NAME, valueSize, value, NULL);
   std::string ret(value);
   free(value);
   return ret;
}

std::string OpenCLAdapter::getDeviceVendor(){
   char* value;
   size_t valueSize;
   clGetDeviceInfo(_dev, CL_DEVICE_VENDOR, 0, NULL, &valueSize);
   value = (char*) malloc(valueSize);
   clGetDeviceInfo(_dev, CL_DEVICE_VENDOR, valueSize, value, NULL);
   std::string ret(value);
   free(value);
   return ret;
}

void OpenCLAdapter::getWorkGroupMultiple(cl_kernel kernel)
{
   clGetKernelWorkGroupInfo(kernel, _dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &_workGroupMultiple, NULL);
}

void OpenCLAdapter::getMaxWorkGroup(cl_kernel kernel)
{
   clGetKernelWorkGroupInfo(kernel, _dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_maxWorkGroup, NULL);
}

//
// OpenCLProcessor implementation.
//

SharedMemAllocator OpenCLProcessor::_shmemAllocator;

OpenCLProcessor::OpenCLProcessor( int devId, memory_space_id_t memId, SMPProcessor *core, SeparateMemoryAddressSpace &mem ) :
   ProcessingElement( &OpenCLDev, memId, 0 /* local node */, 0 /* FIXME: numa */, true, 0 /* socket: n/a? */, false ),
   _core( core ),
   _openclAdapter(),
   _cache( _openclAdapter, this ),
   _devId ( devId ) { }


//TODO: Configure cache awareness
void OpenCLProcessor::initialize()
{
   // Initialize the adapter, it will talk with the OpenCL device.
   _openclAdapter.initialize( OpenCLConfig::getFreeDevice() );

   // Initialize the caching subsystem.
   _cache.initialize();
   
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

BaseThread &OpenCLProcessor::createThread( WorkDescriptor &wd, SMPMultiThread *parent )
{

   OpenCLThread &thr = *NEW OpenCLThread( wd, this, _core );
   thr.setMaxPrefetch( nanos::ext::OpenCLConfig::getPrefetchNum() );

   return thr;
}

void OpenCLProcessor::setKernelBufferArg(void* openclKernel, int argNum, const void* pointer)
{
    cl_mem buffer=_cache.toMemoryObjSS( pointer );
    //Set buffer as arg
    cl_int errCode= clSetKernelArg( (cl_kernel) openclKernel, argNum, sizeof(cl_mem), &buffer ); 
    if( errCode != CL_SUCCESS )
    {
        fatal0kernelNameErr(openclKernel,"Error in setKernelArg with copies/buffer ", errCode);    
    }
}

void OpenCLProcessor::setKernelArg(void* openclKernel, int argNum, size_t size,const void* pointer){
    cl_int errCode= clSetKernelArg( (cl_kernel) openclKernel, argNum, size, pointer );
    if( errCode != CL_SUCCESS )
    {
        if ( errCode == CL_INVALID_ARG_INDEX) {
            fatal0kernelName(openclKernel,"error setting kernel arg, make sure your task declaration"
                    " and OpenCL kernel definition have the same number of arguments");            
        }
        fatal0kernelNameErr(openclKernel,"Error in setKernelArg ", errCode);    
    }
}

void OpenCLProcessor::execKernel(void* openclKernel, 
                        int workDim, 
                        size_t* ndrOffset, 
                        size_t* ndrLocalSize, 
                        size_t* ndrGlobalSize){
    _openclAdapter.execKernel(openclKernel,
                            workDim,
                            ndrOffset,
                            ndrLocalSize,
                            ndrGlobalSize);
}

void OpenCLProcessor::profileKernel(void* openclKernel,
                        int workDim,
						int range_size,
                        size_t* ndrOffset,
                        size_t* ndrLocalSize,
                        size_t* ndrGlobalSize){
    _openclAdapter.profileKernel(openclKernel,
                            workDim,
							range_size,
                            ndrOffset,
                            ndrLocalSize,
                            ndrGlobalSize);
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
   if (_openclAdapter.getUseHostPtr()) {
      message("OpenCL " << _openclAdapter.getDeviceName() << " TRANSFER STATISTICS (using Shared/Mapped memory)");       
   } else {       
      message("OpenCL " << _openclAdapter.getDeviceName() << " TRANSFER STATISTICS");
   }
   message("    Total input transfers: " << bytesToHumanReadable( _cache._bytesIn.value() ) );
   message("    Total output transfers: " << bytesToHumanReadable( _cache._bytesOut.value() ) );
   message("    Total dev2dev(in) transfers: " << bytesToHumanReadable( _cache._bytesDevice.value() ) );
}

void OpenCLProcessor::printProfiling()
{
	_openclAdapter.printProfiling();
}

void OpenCLProcessor::cleanUp()
{
   printStats();
   printProfiling();
}

void* OpenCLProcessor::allocateSharedMemory( size_t size ){    
    return _openclAdapter.allocSharedMemBuffer(size);
}

void OpenCLProcessor::freeSharedMemory( void* addr ){    
    _openclAdapter.freeSharedMemBuffer((void*)(addr));
}

BaseThread &OpenCLProcessor::startOpenCLThread() {
   WD & worker = getWorkerWD();

   NANOS_INSTRUMENT (sys.getInstrumentation()->raiseOpenPtPEvent ( NANOS_WD_DOMAIN, (nanos_event_id_t) worker.getId(), 0, 0 ); )
   NANOS_INSTRUMENT (InstrumentationContextData *icd = worker.getInstrumentationContextData() );
   NANOS_INSTRUMENT (icd->setStartingWD(true) );
   
   _thread=&_core->startThread( *this, worker, NULL );
   
   return *_thread;
}
