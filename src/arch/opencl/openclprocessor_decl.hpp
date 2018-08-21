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

#ifndef _NANOS_OpenCL_PROCESSOR_DECL
#define _NANOS_OpenCL_PROCESSOR_DECL

#include "cachedaccelerator.hpp"
#include "openclcache.hpp"
#include "openclconfig.hpp"
#include "opencldd.hpp"
#include "opencldevice_decl.hpp"
#include "sharedmemallocator.hpp"
#include "smpprocessor.hpp"
#include "openclprofiler.hpp"

#ifdef HAVE_OPENCL_OPENCL_H
#include <OpenCL/opencl.h>
#endif

#ifdef HAVE_CL_OPENCL_H
#include <CL/opencl.h>
#endif

#define MAX_KERNEL_NAME_LENGTH 100

namespace nanos {
namespace ext {

class OpenCLAdapter
{
public: 
   typedef std::map<uint32_t, cl_program> ProgramCache;
   typedef std::map<std::pair<uint64_t,size_t>, cl_mem> BufferCache;

public:
   ~OpenCLAdapter();
   OpenCLAdapter() : _bufCache(), _unmapedCache(), _sizeCache(), _preallocateWholeMemory(false), _synchronize(false), _openCLProfilerDbManager(NULL), _progCache()  {}

public:
   void initialize(cl_device_id dev);

   /**
    * Allocs OpenCL buffer
    * @param size size of the buffer
    * @param host_ptr hostPtr (only used in CPU/mappedMem devices)
    * @param buf buffer object
    * @return return code of OpenCL call
    */
   cl_int allocBuffer( size_t size, void* host_ptr, cl_mem &buf );
   void* allocSharedMemBuffer( size_t size);
   cl_int freeBuffer( cl_mem &buf );
   void freeSharedMemBuffer( void* addr );

   cl_int readBuffer( cl_mem buf, void *dst, size_t offset, size_t size, Atomic<size_t>* globalSizeCounter, cl_event& ev);
   cl_int writeBuffer( cl_mem buf, void *src, size_t offset, size_t size, Atomic<size_t>* globalSizeCounter, cl_event& ev);
   cl_int mapBuffer( cl_mem buf, void *dst, size_t offset, size_t size, cl_event& ev );
   cl_int unmapBuffer( cl_mem buf, void *src, size_t offset, size_t size, cl_event& ev );
   cl_mem getBuffer(SimpleAllocator& allocator, cl_mem parentBuf, uint64_t offset, size_t size );
   size_t getSizeFromCache(uint64_t addr);
   cl_mem createBuffer(cl_mem parentBuf, uint64_t offset, size_t size, void* hostPtr);
   void freeAddr(void* addr );
   cl_int copyInBuffer( cl_mem buf, cl_mem remoteBuffer, size_t offset_buff, size_t offset_remotebuff, size_t size, cl_event& ev );
   
   // Low-level program builder. Lifetime of prog is under caller
   // responsability.
   cl_int buildProgram( const char *src,
                        const char *compilerOpts,
                        cl_program &prog,
                        const std::string& filename );

   // As above, but without compiler options.
   cl_int buildProgram( const char *src, cl_program &prog )
   {
      return buildProgram( src, "", prog, "" );
   }

   // Low-level program destructor.
   cl_int destroyProgram( cl_program &prog );

   // Get program from cache, increasing reference-counting.
   void* getProgram( const char *src,
                      const char *compilerOpts );
   
   void* createKernel(const char* kernel_name, const char* opencl_code, const char* compiler_opts);    

   // Return program to the cache, decreasing reference-counting.
   cl_int putProgram( cl_program &prog );

   void execKernel( void* oclKernel, 
                        int workDim, 
                        size_t* ndrLocalSize, 
                        size_t* ndrGlobalSize);

   /**
    * @brief This function performs kernel executions to determine
    * the best work-group parameters.
    */
   void profileKernel( void* oclKernel,
                       int workDim,
                       size_t* ndrGlobalSize);

   /**
    * @brief This function performs kernel executions to determine
    * the best work-group parameters using the given range.
    */
   void manualProfileKernel( void* oclKernel,
                             std::string kernelName,
                             int workDim,
                             int range_size,
                             const double cost,
                             Dims &dims,
                             size_t* ndrLocalSize,
                             size_t* ndrGlobalSize);

   /**
    * @brief This function performs kernel executions to determine
    * the best work-group parameters using the device hints.
    */
   void automaticProfileKernel(void* oclKernel,
                               std::string kernelName,
                               int workDim,
                               const double cost,
                               Dims &dims,
                               size_t* ndrGlobalSize);


   /**
    * @brief This function returns the OpenCLProfCurrConfig for a given kernel
    * @return NULL whether there is no a execution for this combination or the pointer to the object
    * in the opposite case.
    */
   Execution* getProfStatus(std::string &kernelName, Dims &dims);

   /**
    * @brief Show kernel profiling and information
    */
   void printProfiling();

   // TODO: replace with new APIs.
   size_t getGlobalSize();
   
   std::string getDeviceName();
   std::string getDeviceVendor();
   
   bool outOfOrderSupport(); 

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
   
   bool getUseHostPtr(){
       return _useHostPtrs;
   }
   
   void setPreallocatedWholeMemory(bool val){
      //_preallocateWholeMemory=val;
   }

   ProgramCache& getProgCache() {
      return _progCache;
   }
   
   cl_context& getContext() {
      return _ctx;
   }

   const std::map<std::string, DimsBest>& getBestExec() const {
      return _bestExec;
   }

   const std::map<std::string, DimsExecutions>& getExecutions() const {
      return _nExecutions;
   }

   OpenCLProfilerDbManager* getOpenClProfilerDbManager() const {
      return _openCLProfilerDbManager;
  }

   /**
    * @brief This function update the profiling data during the execution
    */
   void updateProfStats(std::string kernelName, Dims& dims, Execution const &execution);

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

   void setSynchronization( std::string &vendor );

   static inline void clCheckError(cl_int clError, char* errorString) {
      if (clError != CL_SUCCESS) {
         nanos::OpenCLProfilerException(CLP_OPENCL_STANDARD_ERROR, clError, errorString);
      }
   }

   /**
    * @brief Function to launch an OpenCL kernel under profiling mode
    */
   cl_ulong singleExecKernel( void* oclKernel,
                              int workDim,
                              size_t* ndrLocalSize,
                              size_t* ndrGlobalSize);

   /**
    * @brief This function set the work-group multiple preferred values
    */
   size_t getWorkGroupMultiple(cl_kernel kernel);

   /**
    * @brief This function set the maximum work-group on the device
    */
   size_t getMaxWorkGroup(cl_kernel kernel);

private:

   cl_device_id _dev;
   cl_context _ctx;
   cl_command_queue* _queues;
   int _currQueue;
   cl_command_queue _copyInQueue;
   cl_command_queue _copyOutQueue;
   cl_command_queue _profilingQueue;
   BufferCache _bufCache;
   std::map<cl_mem, int> _unmapedCache;
   std::map<uint64_t,size_t> _sizeCache;
   /* >>> OpenCL Profiling >>> */
   std::map<std::string,DimsBest> _bestExec;
   std::map<std::string,DimsExecutions> _nExecutions;
   DevPerfInfo _devPerfInfo;
   /* <<< OpenCL Profiling <<< */
   bool _preallocateWholeMemory;
   bool _synchronize;
   OpenCLProfilerDbManager *_openCLProfilerDbManager;

   ProgramCache _progCache;
   bool _useHostPtrs;
};

class OpenCLProcessor : public ProcessingElement
{
public:        
   OpenCLProcessor( int devId, memory_space_id_t memId, SMPProcessor *core, SeparateMemoryAddressSpace &mem );

   OpenCLProcessor( const OpenCLProcessor &pe ); // Do not implement.
   OpenCLProcessor &operator=( const OpenCLProcessor &pe ); // Do not implement.

public:
    
   void initialize();

   WD &getWorkerWD() const;

   WD &getMasterWD() const;
    
   virtual WD & getMultiWorkerWD () const
   {
      fatal( "getMultiWorkerWD: OpenCLProcessor is not allowed to create MultiThreads" );
   }
   BaseThread & createThread ( WorkDescriptor &wd, SMPMultiThread *parent );
   virtual BaseThread & createMultiThread ( WorkDescriptor &wd, unsigned int numPEs, PE **repPEs )
   {
      fatal( "OpenCLNode is not allowed to create MultiThreads" );
   }
    

   bool supportsUserLevelThreads () const { return false; }

   BaseThread &startOpenCLThread();

   OpenCLAdapter::ProgramCache& getProgCache() {
       return _openclAdapter.getProgCache();
   }
   
   // Get program from cache, increasing reference-counting.
   void* createKernel( const char* kernel_name,
                       const char* opencl_code, 
                       const char* compiler_opts){
       return _openclAdapter.createKernel(kernel_name,opencl_code, compiler_opts);
   }
   
   void setKernelBufferArg(void* openclKernel, int argNum,const void* pointer);
   
   void execKernel(void* openclKernel, 
                        int workDim, 
                        size_t* ndrLocalSize, 
                        size_t* ndrGlobalSize);
   
   void profileKernel(void* openclKernel,
                        int workDim,
                        size_t* ndrGlobalSize);

   void setKernelArg(void* opencl_kernel, int arg_num, size_t size,const void* pointer);
   
   void printStats();

   /**
    * @brief This shows the kernel profiling info and some kernel properties
    */
   void printProfiling();
      
   void cleanUp();
     
   void *allocate( size_t size, uint64_t tag, uint64_t offset )
   {
      return _cache.allocate( size, tag, offset );
   }
   
   void *realloc( void *address, size_t size, size_t ceSize )
   {
      return _cache.reallocate( address, size, ceSize );
   }

   void free( void *address )
   {
      return _cache.free( address );
   }

   bool copyIn( uint64_t devAddr,  uint64_t hostAddr, size_t size, DeviceOps *ops )
   {
      return _cache.copyIn( devAddr, hostAddr, size, ops );
   }

   bool copyOut( uint64_t hostAddr, uint64_t devAddr, size_t size, DeviceOps *ops )
   {
      return _cache.copyOut( hostAddr, devAddr, size, ops );
   }
   
   cl_mem getBuffer( void *localSrc, size_t size )
   {
      return _cache.getBuffer( localSrc, size );
   } 
   
   bool copyInBuffer( void *localSrc, cl_mem remoteBuffer, size_t size, DeviceOps *ops)
   {
      return _cache.copyInBuffer( localSrc, remoteBuffer, size, ops );
   }
   
    cl_context& getContext() {    
        return _openclAdapter.getContext();
    }

    cl_int getOpenCLDeviceType( cl_device_type &deviceType ){
       return _openclAdapter.getDeviceType(deviceType);
    }
    
    static SharedMemAllocator& getSharedMemAllocator() {
       return _shmemAllocator;
    }
   
    void* allocateSharedMemory( size_t size );   
   
    void freeSharedMemory( void* addr );

    
    SimpleAllocator& getCacheAllocator(){
        return _cache.getAllocator();
    }
    
    SimpleAllocator const &getConstCacheAllocator() const {
        return _cache.getConstAllocator();
    }
    
    BaseThread * getOpenCLThread()
    {
       return _thread;
    }


private:
   SMPProcessor *_core;
   BaseThread *_thread;
   OpenCLAdapter _openclAdapter;
   OpenCLCache _cache;
   int _devId;
   static SharedMemAllocator _shmemAllocator;
    

};

} // namespace ext
} // namespace nanos

#endif // _NANOS_OpenCL_PROCESSOR_DECL
