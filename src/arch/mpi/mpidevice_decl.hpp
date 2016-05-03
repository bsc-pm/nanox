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

#ifndef _MPI_DEVICE_DECL
#define _MPI_DEVICE_DECL

#include "workdescriptor_decl.hpp"
#include "processingelement_fwd.hpp"
#include "copydescriptor_decl.hpp"


typedef struct {
       int opId;
       //In case of dev2dev, hostaddr= srcAddr, devAddr=remoteAddr
       uint64_t hostAddr;
       uint64_t devAddr;
       size_t size;
       //size_t old_size;
       //unsigned char* data;
} cacheOrder;

//MPI Communication tags, we use that many so messages don't collide for different operations
enum {
    TAG_M2S_ORDER = 1200, TAG_CACHE_DATA_IN,TAG_CACHE_DATA_OUT, 
    TAG_CACHE_ANSWER, TAG_INI_TASK,TAG_END_TASK, TAG_ENV_STRUCT,TAG_CACHE_ANSWER_REALLOC,
    TAG_CACHE_ANSWER_ALLOC, TAG_CACHE_ANSWER_CIN,TAG_CACHE_ANSWER_COUT,TAG_CACHE_ANSWER_FREE,TAG_CACHE_ANSWER_DEV2DEV,TAG_CACHE_ANSWER_CL,
    TAG_FP_NAME_SYNC, TAG_FP_SIZE_SYNC, TAG_CACHE_DEV2DEV, TAG_EXEC_CONTROL, TAG_NUM_PENDING_COMMS, TAG_UNIFIED_MEM
};

//Because of DEV2DEV OPIDs <=0 are RESERVED, and OPIDs > OPID_DEVTODEV too
enum {
    OPID_FINISH=1, OPID_COPYIN = 2, OPID_COPYOUT=3, OPID_FREE = 4, OPID_ALLOCATE =5 , OPID_COPYLOCAL = 6, OPID_REALLOC = 7, OPID_CONTROL = 8, 
    OPID_CREATEAUXTHREAD=9, OPID_UNIFIED_MEM_REQ=10, OPID_TASK_INIT=11, /*Keep DEV2DEV value as highest in the OPIDs*/ OPID_DEVTODEV=999
};
//Assigned rank value for the Daemon Thread, so it doesn't get used by any DD
#define CACHETHREADRANK -1
#define TASK_END_PROCESS -1
//When source or destination comes with this value, it means that the user
//didn't specify any concrete device, runtime launchs in whatever it wants
//so we have to override it's value with the PE value
//WARNING: Keep this defines with the same value than the one existing in the compiler (nanox-mpi.hpp)
#define UNKOWN_RANKSRCDST -95
#define MASK_TASK_NUMBER 989



namespace nanos {

/* \brief Device specialization for MPI architecture
 * provides functions to allocate and copy data in the device
 */
   class MPIDevice : public Device
   {
      private:
         static char _executingTask;
         //static Directory *_masterDir;
         static bool _createdExtraWorkerThread;

         static void getMemoryLockLimit();
         
         /*! \brief MPIDevice copy constructor
          */
         explicit MPIDevice ( const MPIDevice &arch );

      public:
          
         static MPI_Datatype cacheStruct;    
         /*! \brief MPIDevice constructor
          */
         MPIDevice ( const char *n );

         /*! \brief MPIDevice destructor
          */
         ~MPIDevice(); 

         /* \brief Reallocate and copy from address
          */
         static void * realloc( void * address, size_t size, size_t ceSize, ProcessingElement *pe );
         
         /**
          * Initialice cache struct datatype MPI so we can use it
          */
         static void initMPICacheStruct();
         
         static void remoteNodeCacheWorker();
         
         
         virtual void *memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem, WD const *wd, unsigned int copyIdx);
         virtual void memFree( uint64_t addr, SeparateMemoryAddressSpace &mem );
         virtual void _canAllocate( SeparateMemoryAddressSpace &mem, std::size_t *sizes, unsigned int numChunks, std::size_t *remainingSizes ) { }
         virtual std::size_t getMemCapacity( SeparateMemoryAddressSpace &mem );

         virtual void _copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
         virtual void _copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
         virtual bool _copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
         virtual void _copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace const &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) { fatal0("Strided copies not implemented if offload");}
         virtual void _copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) { fatal0("Strided copies not implemented if offload");}
         virtual bool _copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace const &memDest, SeparateMemoryAddressSpace const &memOrig, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) { fatal0("Strided copies not implemented if offload"); }
         virtual void _getFreeMemoryChunksList( SeparateMemoryAddressSpace const &mem, SimpleAllocator::ChunkList &list ) const { fatal0(__PRETTY_FUNCTION__ << " not implemented if offload"); }

   };
} // namespace nanos

#endif
