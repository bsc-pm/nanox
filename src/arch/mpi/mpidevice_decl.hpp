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

#ifndef _MPI_DEVICE_DECL
#define _MPI_DEVICE_DECL

#include "workdescriptor_decl.hpp"
#include "processingelement_fwd.hpp"
#include "copydescriptor_decl.hpp"



typedef struct {
       short opId;
       uint64_t hostAddr;
       uint64_t devAddr;
       size_t size;
       size_t old_size;
       //unsigned char* data;
} cacheOrder;

//MPI Communication tags, we use that many so messages don't collide for different operations
enum {
    TAG_CACHE_ORDER = 200, TAG_CACHE_DATA_IN =201,TAG_CACHE_DATA_OUT =205, 
    TAG_CACHE_ANSWER =202, TAG_INI_TASK=99,TAG_END_TASK=100, TAG_ENV_STRUCT=101,TAG_CACHE_ANSWER_REALLOC,
    TAG_CACHE_ANSWER_ALLOC, TAG_CACHE_ANSWER_CIN,TAG_CACHE_ANSWER_COUT,TAG_CACHE_ANSWER_FREE,TAG_CACHE_ANSWER_CL
};

enum {
    OPID_FINISH=-1, OPID_COPYIN = 1, OPID_COPYOUT=2, OPID_FREE = 3, OPID_ALLOCATE =4 , OPID_COPYLOCAL = 5, OPID_REALLOC = 6
};
//Assigned rank value for the Daemon Thread, so it doesn't get used by any DD
#define CACHETHREADRANK -1
//When source or destination comes with this value, it means that the user
//didn't specify any concrete device, runtime launchs in whatever it wants
//so we have to override it's value with the PE value
#define UNKOWN_RANKSRCDST -2



namespace nanos
{

/* \brief Device specialization for MPI architecture
 * provides functions to allocate and copy data in the device
 */
   class MPIDevice : public Device
   {
      private:
         static unsigned int _rlimit;
         static Directory *_masterDir;

         static void getMemoryLockLimit();

         /*! \brief copy in when the thread invoking this function belongs to pe
//          */
//         static bool isMycopyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size, ProcessingElement *pe );
//
//         /*! \brief copy in when the thread invoking this function does not belong to pe
//          *         In this case, the information about the copy is added to a list, and the appropriate
//          *         thread (which is periodically checking the list) will perform the copy and notify
//          *         the cache when it has finished
//          */
//         static bool isNotMycopyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size, ProcessingElement *pe );
//
//         /*! \brief copy out when the thread invoking this function belongs to pe
//          */
//         static bool isMycopyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size, ProcessingElement *pe );
//
//         /*! \brief copy out when the thread invoking this function does not belong to pe
//          *         In this case, the information about the copy is added to a list, and the appropriate
//          *         thread (which is periodically checking the list) will perform the copy and notify
//          *         the cache when it has finished
//          */
//         static bool isNotMycopyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size, ProcessingElement *pe );

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

         /* \brief allocate the whole memory of the MPI device
          *        If the allocation fails due to a CUDA memory-related error,
          *        this function keeps trying to allocate as much memory as
          *        possible by trying smaller sizes from 100% to 50%, decrementing
          *        by 5% each time
          *        On success, returns a pointer to the allocated memory and rewrites
          *        size with the final amount of allocated memory
          */
//         static void * allocateWholeMemory( size_t &size );
//
//         /* \brief free the whole MPI device memory pointed by address
//          */
//         static void freeWholeMemory( void * address );
//
//         /* \brief allocate a chunk of pinned memory of the host
//          */
//         static void * allocatePinnedMemory( size_t size );
//
//         /* \brief free the chunk of pinned host memory pointed by address
//          */
//         static void freePinnedMemory( void * address );

         /* \brief allocate size bytes in the device
          */
         static void * allocate( size_t size, ProcessingElement *pe );

         /* \brief free address
          */
         static void free( void *address, ProcessingElement *pe );

         /* \brief Copy from remoteSrc in the host to localDst in the device
          *        Returns true if the operation is synchronous
          */
         static bool copyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size, ProcessingElement *pe );

         /* \brief Copy from localSrc in the device to remoteDst in the host
          *        Returns true if the operation is synchronous
          */
         static bool copyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size, ProcessingElement *pe );

         /* \brief Copy locally in the device from src to dst
          */
         static void copyLocal( void *dst, void *src, size_t size, ProcessingElement *pe );

         /* \brief When using asynchronous transfer modes, this function is used to notify
          *        the PE that another MPI has requested the data synchronization related to
          *        hostAddress
          */
         static void syncTransfer( uint64_t hostAddress, ProcessingElement *pe);

         /* \brief Reallocate and copy from address
          */
         static void * realloc( void * address, size_t size, size_t ceSize, ProcessingElement *pe );
         
         /**
          * Initialice cache struct datatype MPI so we can use it
          */
         static void initMPICacheStruct();
         
         static void mpiCacheWorker();
         
         static void setMasterDirectory(Directory *dir) {_masterDir=dir;};


         /* \brief copy from src in the host to dst in the device synchronously
          */
         //static void copyInSyncToDevice ( void * dst, void * src, size_t size );

         /* \brief when transferring with asynchronous modes, copy from src in the host
          *        to dst in the host, where dst is an intermediate buffer
          */
         //static void copyInAsyncToBuffer( void * dst, void * src, size_t size );

         /* \brief when transferring with asynchronous modes, copy from src in the host
          *        to dst in the device, where src is an intermediate buffer
          */
         //static void copyInAsyncToDevice( void * dst, void * src, size_t size );

         /* \brief when transferring with asynchronous modes, wait until all input copies
          *        (from host to device) have been completed
          */
         //static void copyInAsyncWait();

         /* \brief when transferring with synchronous mode, copy from src in the device
          *        to dst in the host
          */
         //static void copyOutSyncToHost ( void * dst, void * src, size_t size );

         /* \brief when transferring with asynchronous modes, copy from src in the device
          *        to dst in the host, where dst is an intermediate buffer
          */
         //static void copyOutAsyncToBuffer( void * src, void * dst, size_t size );

         /* \brief when transferring with asynchronous modes, wait until all output copies
          *        (from device to host) have been completed
          */
        // static void copyOutAsyncWait();

         /* \brief when transferring with asynchronous modes, copy from src in the host
          *        to dst in the host, where src is an intermediate buffer
          */
         //static void copyOutAsyncToHost( void * src, void * dst, size_t size );

         /* \brief Copy from addrSrc in peSrc device to addrDst in peDst device
          *        Returns true if the operation is synchronous
          */
         static bool copyDevToDev( void * addrDst, CopyDescriptor& dstCd, void * addrSrc, std::size_t size, ProcessingElement *peDst, ProcessingElement *peSrc );
   };
}

#endif
