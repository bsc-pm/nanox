/*************************************************************************************/
/*      Copyright 2013 Barcelona Supercomputing Center                               */
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

#ifndef _NANOS_OpenCL_PROCESSOR_DECL
#define _NANOS_OpenCL_PROCESSOR_DECL

#include "cachedaccelerator.hpp"
#include "openclcache.hpp"
#include "openclconfig.hpp"
#include "opencldd.hpp"
#include "opencldevice_decl.hpp"
#include "sharedmemallocator.hpp"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define MAX_KERNEL_NAME_LENGTH 100

namespace nanos {
namespace ext {

class OpenCLAdapter
{
public: 
   typedef std::map<uint32_t, cl_program> ProgramCache;

public:
   ~OpenCLAdapter();
   OpenCLAdapter() : _preallocateWholeMemory(false){}

public:
   void initialize(cl_device_id dev);

   cl_int allocBuffer( size_t size, cl_mem &buf );
   void* allocSharedMemBuffer( size_t size);
   cl_int freeBuffer( cl_mem &buf );
   void freeSharedMemBuffer( void* addr );

   cl_int readBuffer( cl_mem buf, void *dst, size_t offset, size_t size );
   cl_int writeBuffer( cl_mem buf, void *src, size_t offset, size_t size );
   cl_int mapBuffer( cl_mem buf, void *dst, size_t offset, size_t size );
   cl_int unmapBuffer( cl_mem buf, void *src, size_t offset, size_t size );
   cl_mem getBuffer( cl_mem parentBuf, size_t offset, size_t size );
   void freeAddr( void* addr );
   cl_int copyInBuffer( cl_mem buf, cl_mem remoteBuffer, size_t offset_buff, size_t offset_remotebuff, size_t size );

   // Low-level program builder. Lifetime of prog is under caller
   // responsability.
   cl_int buildProgram( const char *src,
                        const char *compilerOpts,
                        cl_program &prog );

   // As above, but without compiler options.
   cl_int buildProgram( const char *src, cl_program &prog )
   {
      return buildProgram( src, "", prog );
   }

   // Low-level program destructor.
   cl_int destroyProgram( cl_program &prog );

   // Get program from cache, increasing reference-counting.
   void* getProgram( const char *src,
                      const char *compilerOpts );
   
   void* createKernel(const char* kernel_name, const char* opencl_code, const char* compiler_opts);    

   // Return program to the cache, decreasing reference-counting.
   cl_int putProgram( cl_program &prog );

   cl_int execKernel( void* oclKernel, 
                        int workDim, 
                        size_t* ndrOffset, 
                        size_t* ndrLocalSize, 
                        size_t* ndrGlobalSize);

   // TODO: replace with new APIs.
   size_t getGlobalSize();
   
   void waitForEvents();
   
   

public:
   cl_int getDeviceType( cl_device_type &deviceType )
   {
      return getDeviceInfo( CL_DEVICE_TYPE,
                            sizeof( cl_device_type ),
                            &deviceType );
   }


   cl_int getSizeTypeMax( unsigned long long &sizeTypeMax );

   cl_int getPreferredWorkGroupSizeMultiple( size_t &preferredWorkGroupSizeMultiple );
   
   bool getPreallocatesWholeMemory(){
       return _preallocateWholeMemory;
   }
   
   void setPreallocatedWholeMemory(bool val){
      // _preallocateWholeMemory=val;
   }

   ProgramCache& getProgCache() {
        return _progCache;
    }
   
   cl_context& getContext() {
        return _ctx;
    }
   
   cl_command_queue& getCommandQueue(){
       return _queue;
   }

private:
   cl_int getDeviceInfo( cl_device_info key, size_t size, void *value );

   cl_int
   getStandardPreferredWorkGroupSizeMultiple(
      size_t &preferredWorkGroupSizeMultiple );
   cl_int
   getNVIDIAPreferredWorkGroupSizeMultiple(
      size_t &preferredWorkGroupSizeMultiple );

   cl_int getStandardSizeTypeMax( unsigned long long &sizeTypeMax );
   cl_int getNVIDIASizeTypeMax( unsigned long long &sizeTypeMax );

   cl_int getPlatformName( std::string &name );

private:
   cl_device_id _dev;
   cl_context _ctx;
   cl_command_queue _queue;
   std::map<void *, cl_mem> _bufCache;
   const bool _preallocateWholeMemory;

   ProgramCache _progCache;
   std::vector<cl_event> _pendingEvents;
};

class OpenCLProcessor : public CachedAccelerator<OpenCLDevice>
{
public:        
   OpenCLProcessor( int id , int devId, int uid );

   OpenCLProcessor( const OpenCLProcessor &pe ); // Do not implement.
   OpenCLProcessor &operator=( const OpenCLProcessor &pe ); // Do not implement.

public:
    
   void initialize();

   WD &getWorkerWD() const;

   WD &getMasterWD() const;
   
   BaseThread &createThread( WorkDescriptor &wd );

   bool supportsUserLevelThreads() const { return false; }
    
   OpenCLAdapter::ProgramCache& getProgCache() {
       return _openclAdapter.getProgCache();
   }
   
   // Get program from cache, increasing reference-counting.
   void* createKernel( const char* kernel_name,
                       const char* opencl_code, 
                       const char* compiler_opts){
       return _openclAdapter.createKernel(kernel_name,opencl_code, compiler_opts);
   }
   
   void setKernelBufferArg(void* openclKernel, int argNum, void* pointer);
   
   void execKernel(void* openclKernel, 
                        int workDim, 
                        size_t* ndrOffset, 
                        size_t* ndrLocalSize, 
                        size_t* ndrGlobalSize);
   
   void setKernelArg(void* opencl_kernel, int arg_num, size_t size, void* pointer);
   
   void printStats();
   
   void waitForEvents() {       
       _openclAdapter.waitForEvents();
   }
   
   void cleanUp();
     
   void *allocate( size_t size, uint64_t tag )
   {
      return _cache.allocate( size, tag );
   }
   
   void *realloc( void *address, size_t size, size_t ceSize )
   {
      return _cache.reallocate( address, size, ceSize );
   }

   void free( void *address )
   {
      return _cache.free( address );
   }

   bool copyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size )
   {
      return _cache.copyIn( localDst, remoteSrc, size );
   }

   bool copyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size )
   {
      return _cache.copyOut( remoteDst, localSrc, size );
   }
   
   cl_mem getBuffer( void *localSrc, size_t size )
   {
      return _cache.getBuffer( localSrc, size );
   } 
   
   bool copyInBuffer( void *localSrc, cl_mem remoteBuffer, size_t size )
   {
      return _cache.copyInBuffer( localSrc, remoteBuffer, size );
   }
   
    cl_context& getContext() {    
        return _openclAdapter.getContext();
    }

    cl_command_queue& getCommandQueue() {    
        return _openclAdapter.getCommandQueue();
    }

    cl_int getOpenCLDeviceType( cl_device_type &deviceType ){
       return _openclAdapter.getDeviceType(deviceType);
    }
   
    static SharedMemAllocator& getSharedMemAllocator() {
       return _shmemAllocator;
    }
   
    void* allocateSharedMemory( size_t size );   
   
    void freeSharedMemory( void* addr );


private:
   OpenCLAdapter _openclAdapter;
   OpenCLCache _cache;
   int _devId;
   static SharedMemAllocator _shmemAllocator;
    

};

} // End namespace ext.
} // End namespace nanos.

#endif // _NANOS_OpenCL_PROCESSOR_DECL
