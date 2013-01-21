/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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

#ifndef _NANOS_OCL_PROCESSOR_DECL
#define _NANOS_OCL_PROCESSOR_DECL

#include "cachedaccelerator.hpp"
#include "oclcache.hpp"
#include "oclconfig.hpp"
#include "ocldd.hpp"
#include "ocldevice_decl.hpp"

namespace nanos {
namespace ext {

class OCLAdapter
{
public: 
   typedef std::map<uint32_t, cl_program> ProgramCache;

public:
   ~OCLAdapter();

public:
   void initialize(cl_device_id dev);

   cl_int allocBuffer( size_t size, cl_mem &buf );
   cl_int freeBuffer( cl_mem &buf );

   cl_int readBuffer( cl_mem buf, void *dst, size_t offset, size_t size );
   cl_int writeBuffer( cl_mem buf, void *src, size_t offset, size_t size );

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
   
   void* createKernel( char* kernel_name, void* program);    

   // Return program to the cache, decreasing reference-counting.
   cl_int putProgram( cl_program &prog );

   cl_int execKernel( void* openclKernel, 
                        int workDim, 
                        size_t* ndrOffset, 
                        size_t* ndrLocalSize, 
                        size_t* ndrGlobalSize);

   // TODO: replace with new APIs.
   size_t getGlobalSize();
   
   

public:
   cl_int getDeviceType( unsigned long long &deviceType )
   {
      return getDeviceInfo( CL_DEVICE_TYPE,
                            sizeof( unsigned long long ),
                            &deviceType );
   }

   cl_int
   getMaxComputeUnits( unsigned &maxComputeUnits )
   {
      return getDeviceInfo( CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof( unsigned ),
                            &maxComputeUnits );
   }

   cl_int getMaxWorkItemDimensions( unsigned &maxWorkItemDimensions )
   {
      return getDeviceInfo( CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            sizeof( unsigned ),
                            &maxWorkItemDimensions );
   }

   cl_int getMaxWorkGroupSize( size_t &maxWorkGroupSize )
   {
      return getDeviceInfo( CL_DEVICE_MAX_WORK_GROUP_SIZE,
                            sizeof( size_t ),
                            &maxWorkGroupSize );
   }

   cl_int getMaxMemoryAllocSize( size_t &maxMemoryAllocSize )
   {
      return getDeviceInfo( CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                            sizeof( size_t ),
                            &maxMemoryAllocSize );
   }

   cl_int getLocalMemoryMapping( unsigned &localMemoryMapping )
   {
      return getDeviceInfo( CL_DEVICE_LOCAL_MEM_TYPE,
                            sizeof( unsigned ),
                            &localMemoryMapping );
   }

   cl_int getLocalMemorySize( size_t &localMemorySize )
   {
      return getDeviceInfo( CL_DEVICE_LOCAL_MEM_SIZE,
                            sizeof( size_t ),
                            &localMemorySize );
   }

   cl_int
   getSupportErrorCorrection( unsigned long long &supportErrorCorrection )
   {
      return getDeviceInfo( CL_DEVICE_ERROR_CORRECTION_SUPPORT,
                            sizeof( unsigned long long ),
                            &supportErrorCorrection );
   }

   cl_int getSizeTypeMax( unsigned long long &sizeTypeMax );

   cl_int getPreferredWorkGroupSizeMultiple( size_t &preferredWorkGroupSizeMultiple );

   ProgramCache getProgCache() const {
        return _progCache;
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

   ProgramCache _progCache;
};

class OCLProcessor : public CachedAccelerator<OCLDevice>
{
public:
   OCLProcessor( int id );

   OCLProcessor( const OCLProcessor &pe ); // Do not implement.
   OCLProcessor &operator=( const OCLProcessor &pe ); // Do not implement.

public:
   void initialize();

   WD &getWorkerWD() const;

   WD &getMasterWD() const;

   BaseThread &createThread( WorkDescriptor &wd );

   bool supportsUserLevelThreads() const { return false; }
    
   OCLAdapter::ProgramCache getProgCache() const {
        return _oclAdapter.getProgCache();
   }
   
   // Get program from cache, increasing reference-counting.
   void* getProgram( const char *src,
                      const char *compilerOpts ){
       return _oclAdapter.getProgram(src,compilerOpts);
   }
   
   // Get program from cache, increasing reference-counting.
   void* createKernel( char* kernel_name,
                       void* program){
       return _oclAdapter.createKernel(kernel_name,program);
   }
   
   void setKernelBufferArg(void* oclKernel, int argNum, void* pointer);
   
   void execKernel(void* oclKernel, 
                        int workDim, 
                        size_t* ndrOffset, 
                        size_t* ndrLocalSize, 
                        size_t* ndrGlobalSize);
   
   void setKernelArg(void* ocl_kernel, int arg_num, size_t size, void* pointer){
        p_clSetKernelArg( (cl_kernel) ocl_kernel, arg_num, size, pointer );
   }
   
   void *allocate( size_t size )
   {
      return _cache.allocate( size );
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

   bool asyncCopyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size )
   {
      return _dma.copyIn( localDst, remoteSrc, size );
   }

   bool asyncCopyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size )
   {
      return _dma.copyOut( remoteDst, localSrc, size );
   }

   void syncTransfer( uint64_t hostAddress )
   {
      _dma.syncTransfer( hostAddress );
   }

   void execTransfers()
   {
      _dma.execTransfers();
   }



private:
   OCLAdapter _oclAdapter;
   OCLCache _cache;
   OCLDMA _dma;

};

}
 // End namespace ext.
} // End namespace nanos.

#endif // _NANOS_OCL_PROCESSOR_DECL