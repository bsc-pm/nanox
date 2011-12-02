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


#ifndef _GASNET_API
#define _GASNET_API

#include "network_decl.hpp"
#include "networkapi.hpp"
#include "simpleallocator.hpp"
#include "directory.hpp"
#include "clusterinfo_decl.hpp"
//#include "remoteworkgroup_decl.hpp"
#include <map>

extern "C" {
#include <gasnet.h>
}

namespace nanos {
namespace ext {

   class GASNetAPI : public NetworkAPI
   {
      private:
         static Network *_net;

         static RemoteWorkGroup *_rwgGPU;
         static RemoteWorkGroup *_rwgSMP;
         static Directory *_masterDir;
#ifndef GASNET_SEGMENT_EVERYTHING
         static SimpleAllocator *_thisNodeSegment;
#endif
         static std::set< void * > _waitingPutRequests;
         static std::set< void * > _receivedUnmatchedPutRequests;
         static Lock _waitingPutRequestsLock;
         static std::vector< SimpleAllocator * > _pinnedAllocators;
         static std::vector< Lock * > _pinnedAllocatorsLocks;
         static Atomic<unsigned int> *_seqN;

         // data dependencies for data comming via a node different than the one sending the work
         // e.g. node 1 sends work to 2 but some data comes from node 3, the work message can
         // arrive from 1 to 2 before than the data from 3 to 2. 
         typedef struct {
            WD *wd;
            unsigned int count;
         } wdDeps;
         static Lock _depsLock;
         static Lock _sentDataLock;
         static std::vector<std::set<uint64_t> *> _sentData;
         static std::multimap<uint64_t, wdDeps *> _depsMap; //needed data
         static std::set<uint64_t> _recvdDeps; //already got the data

         // GASNet does not allow to send a message during the execution of an active message handler,
         // to handle put request, we have to enqueue the incoming request to be able to send the "real" put
         // later on (we do it during the GASNetAPI::poll).
         struct putReqDesc {
            unsigned int dest;
            void *origAddr;
            void *destAddr;
            std::size_t len;
            void *tmpBuffer;
         };
         static std::list<struct putReqDesc * > _putReqs;
         static Lock _putReqsLock;
         static std::size_t rxBytes;
         static std::size_t txBytes;
         static std::size_t _totalBytes;
         
         static std::list< void * > _freeBufferReqs;
         static Lock _freeBufferReqsLock;
         static std::list< std::pair<void *, unsigned int> > _workDoneReqs;
         static Lock _workDoneReqsLock;
         static std::list< std::pair< unsigned int, std::pair<WD *, std::vector<uint64_t> *> > > _deferredWorkReqs;
         static Lock _deferredWorkReqsLock;
         static Atomic<unsigned int> _recvSeqN;
         
      public:
         void initialize ( Network *net );
         void finalize ();
         void poll ();
         void sendExitMsg ( unsigned int dest );
         void sendWorkMsg ( unsigned int dest, void ( *work ) ( void * ), unsigned int arg0, unsigned int arg1, unsigned int numPe, size_t argSize, char * arg, void ( *xlate ) ( void *, void * ), int arch, void *wd );
         void sendWorkDoneMsg ( unsigned int dest, void *remoteWdAddr, int peId);
         static void _sendWorkDoneMsg ( unsigned int dest, void *remoteWdAddr, int peId);
         void put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size );
         void get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, size_t size );
         void malloc ( unsigned int remoteNode, size_t size, void *waitObjAddr );
         void memFree ( unsigned int remoteNode, void *addr );
         void memRealloc ( unsigned int remoteNode, void *oldAddr, size_t oldSize, void *newAddr, size_t newSize );
         void nodeBarrier( void );
         
         void sendMyHostName( unsigned int dest );
         void sendRequestPut( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, size_t len );
         void setMasterDirectory(Directory *dir);
         std::size_t getTotalBytes();
         static void testForDependencies( WD *localWD, std::vector<uint64_t> *deps );
         static std::size_t getRxBytes();
         static std::size_t getTxBytes();

      private:
         void _put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size, void *remoteTmpBuffer );
         static void enqueuePutReq( unsigned int dest, void *origAddr, void *destAddr, std::size_t len, void *tmpBuffer );
         void checkForPutReqs();
         static void sendWaitForRequestPut( unsigned int dest, uint64_t addr );
         static void print_copies( WD *wd, int deps );
         static void releaseWDsFromDataDep( void *addr );
         static void enqueueFreeBufferNotify( void *bufferAddr );
         static void checkForFreeBufferReqs();
         static void checkWorkDoneReqs();
         static void checkDeferredWorkReqs();
         static void sendFreeTmpBuffer( void *addr );

         // Active Message handlers
         static void amFinalize( gasnet_token_t token );
         static void amFinalizeReply(gasnet_token_t token);
         static void amWork(gasnet_token_t token, void *arg, std::size_t argSize,
                             gasnet_handlerarg_t workLo,
                             gasnet_handlerarg_t workHi,
                             gasnet_handlerarg_t xlateLo,
                             gasnet_handlerarg_t xlateHi,
                             gasnet_handlerarg_t rmwdLo,
                             gasnet_handlerarg_t rmwdHi,
                             unsigned int dataSize, unsigned int wdId, unsigned int numPe, int arch, unsigned int seq );
         static void amWorkData(gasnet_token_t token, void *buff, std::size_t len,
               gasnet_handlerarg_t msgNum,
               gasnet_handlerarg_t totalLenLo,
               gasnet_handlerarg_t totalLenHi);
                  
         static void amWorkDone( gasnet_token_t token, gasnet_handlerarg_t addrLo, gasnet_handlerarg_t addrHi, gasnet_handlerarg_t peId );
         static void amMalloc( gasnet_token_t token, gasnet_handlerarg_t sizeLo, gasnet_handlerarg_t sizeHi,
                                gasnet_handlerarg_t waitObjAddrLo, gasnet_handlerarg_t waitObjAddrHi );
         static void amMallocReply( gasnet_token_t token, gasnet_handlerarg_t addrLo, gasnet_handlerarg_t addrHi,
                                      gasnet_handlerarg_t waitObjAddrLo, gasnet_handlerarg_t waitObjAddrHi );
         static void amFree( gasnet_token_t token, gasnet_handlerarg_t addrLo, gasnet_handlerarg_t addrHi );
         static void amRealloc( gasnet_token_t token, gasnet_handlerarg_t oldAddrLo, gasnet_handlerarg_t oldAddrHi,
                                    gasnet_handlerarg_t oldSizeLo, gasnet_handlerarg_t oldSizeHi,
                                    gasnet_handlerarg_t newAddrLo, gasnet_handlerarg_t newAddrHi,
                                    gasnet_handlerarg_t newSizeLo, gasnet_handlerarg_t newSizeHi);
         static void amMasterHostname( gasnet_token_t token, void *buff, std::size_t nbytes );
         static void amPut( gasnet_token_t token,
               void *buf,
               std::size_t len,
               gasnet_handlerarg_t origAddrLo,
               gasnet_handlerarg_t origAddrHi,
               gasnet_handlerarg_t tagAddrLo,
               gasnet_handlerarg_t tagAddrHi,
               gasnet_handlerarg_t first,
               gasnet_handlerarg_t last);
         static void amGet( gasnet_token_t token,
               gasnet_handlerarg_t destAddrLo,
               gasnet_handlerarg_t destAddrHi,
               gasnet_handlerarg_t origAddrLo,
               gasnet_handlerarg_t origAddrHi,
               gasnet_handlerarg_t tagAddrLo,
               gasnet_handlerarg_t tagAddrHi,
               gasnet_handlerarg_t lenLo,
               gasnet_handlerarg_t lenHi,
               gasnet_handlerarg_t waitObjLo,
               gasnet_handlerarg_t waitObjHi );
         static void amGetReply( gasnet_token_t token,
               void *buf,
               std::size_t len,
               gasnet_handlerarg_t waitObjLo,
               gasnet_handlerarg_t waitObjHi);
         static void amPutF( gasnet_token_t token,
               gasnet_handlerarg_t destAddrLo,
               gasnet_handlerarg_t destAddrHi,
               gasnet_handlerarg_t len,
               gasnet_handlerarg_t wordSize,
               gasnet_handlerarg_t valueLo,
               gasnet_handlerarg_t valueHi );
         static void amRequestPut( gasnet_token_t token,
               gasnet_handlerarg_t destAddrLo,
               gasnet_handlerarg_t destAddrHi,
               gasnet_handlerarg_t origAddrLo,
               gasnet_handlerarg_t origAddrHi,
               gasnet_handlerarg_t tmpBufferLo,
               gasnet_handlerarg_t tmpBufferHi,
               gasnet_handlerarg_t lenLo,
               gasnet_handlerarg_t lenHi,
               gasnet_handlerarg_t dst );
         static void amWaitRequestPut( gasnet_token_t token, 
               gasnet_handlerarg_t addrLo,
               gasnet_handlerarg_t addrHi );
         static void amFreeTmpBuffer( gasnet_token_t token, 
               gasnet_handlerarg_t addrLo,
               gasnet_handlerarg_t addrHi );
   };
}
}
#endif
