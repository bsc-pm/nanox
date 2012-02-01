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

#include "gasnetapi_decl.hpp"
#include "smpdd.hpp"

#ifdef GPU_DEV
//FIXME: GPU Support
#include "gpudd.hpp"
#include "gpudevice_decl.hpp"
#endif

#include "system.hpp"
#include "os.hpp"
#include "clusterinfo_decl.hpp"
#include "instrumentation.hpp"
#include "atomic_decl.hpp"
#include <list>
#include <cstddef>

//#define HALF_PRESEND

#ifdef HALF_PRESEND
Atomic<int> wdindc = 0;
WD* buffWD = NULL;
#endif

typedef struct {
   void (*outline) (void *);
} nanos_smp_args_t;

#ifdef GPU_DEV
//FIXME: GPU Support
void * local_nanos_gpu_factory( void *prealloc, void *args );
void * local_nanos_gpu_factory( void *prealloc, void *args )
{
   nanos_smp_args_t *smp = ( nanos_smp_args_t * ) args;
   if ( prealloc != NULL )
   {
      return ( void * )new (prealloc) ext::GPUDD( smp->outline );
   }
   else
   {
      return ( void * ) new ext::GPUDD( smp->outline );
   }
}
#endif
void * local_nanos_smp_factory( void *prealloc, void *args );
void * local_nanos_smp_factory( void *prealloc, void *args )
{
   nanos_smp_args_t *smp = ( nanos_smp_args_t * ) args;

   if ( prealloc != NULL )
   {
      return ( void * )new (prealloc) ext::SMPDD( smp->outline );
   }
   else 
   {
      return ( void * )new ext::SMPDD( smp->outline );
   }
}

#ifndef __SIZEOF_POINTER__
#   error This compiler does not define __SIZEOF_POINTER__ :( 
#else
#   if __SIZEOF_POINTER__ == 8
#      define MERGE_ARG( _Hi, _Lo) (  ( uint32_t ) _Lo + ( ( ( uintptr_t ) ( ( uint32_t ) _Hi ) ) << 32 ) )
#      define ARG_HI( _Arg ) ( ( gasnet_handlerarg_t ) ( ( ( uintptr_t ) ( _Arg ) ) >> 32 ) )
#      define ARG_LO( _Arg ) ( ( gasnet_handlerarg_t ) ( ( uintptr_t ) _Arg ) )
#   else
#      define MERGE_ARG( _Hi, _Lo) ( ( uintptr_t ) ( _Lo ) )
#      define ARG_HI( _Arg ) ( ( gasnet_handlerarg_t ) 0 )
#      define ARG_LO( _Arg ) ( ( gasnet_handlerarg_t ) _Arg )
#   endif
#endif

using namespace nanos;
using namespace ext;

Network * GASNetAPI::_net;

RemoteWorkGroup * GASNetAPI::_rwgGPU;
RemoteWorkGroup * GASNetAPI::_rwgSMP;
Directory * GASNetAPI::_masterDir;
#ifndef GASNET_SEGMENT_EVERYTHING
SimpleAllocator * GASNetAPI::_thisNodeSegment;
#endif

Lock GASNetAPI::_depsLock;
Lock GASNetAPI::_sentDataLock;
std::vector<std::set<uint64_t> *> GASNetAPI::_sentData;
std::multimap<uint64_t, GASNetAPI::wdDeps *> GASNetAPI::_depsMap; //needed data
std::set<uint64_t> GASNetAPI::_recvdDeps; //already got the data

std::list<struct GASNetAPI::putReqDesc * > GASNetAPI::_putReqs;
Lock GASNetAPI::_putReqsLock;
std::set<void *> GASNetAPI::_receivedUnmatchedPutRequests;
std::set<void *> GASNetAPI::_waitingPutRequests;
Lock GASNetAPI::_waitingPutRequestsLock;
std::size_t GASNetAPI::rxBytes = 0;
std::size_t GASNetAPI::txBytes = 0;
std::size_t GASNetAPI::_totalBytes = 0;
std::vector< SimpleAllocator * > GASNetAPI::_pinnedAllocators;
std::vector< Lock * > GASNetAPI::_pinnedAllocatorsLocks;
std::list< void * > GASNetAPI::_freeBufferReqs;
Lock GASNetAPI::_freeBufferReqsLock;

Lock GASNetAPI::_workDoneReqsLock;
std::list< std::pair<void *, unsigned int> > GASNetAPI::_workDoneReqs;
Atomic<unsigned int> *GASNetAPI::_seqN;
Atomic<unsigned int> GASNetAPI::_recvSeqN (0);
Lock GASNetAPI::_deferredWorkReqsLock;
std::list< std::pair< unsigned int, std::pair<WD *, std::vector<uint64_t> *> > > GASNetAPI::_deferredWorkReqs;
#if 0
extern char **environ;
static void inspect_environ(void)
{
   int i = 0;

   fprintf(stderr, "+------------- Environ Start = %p --------------\n", environ);
   while (environ[i] != NULL)
      fprintf(stderr, "| %s\n", environ[i++]);
   fprintf(stderr, "+-------------- Environ End = %p ---------------\n", &environ[i]);
}
#endif

void GASNetAPI::print_copies( WD *wd, int deps )
{
#if 1
   unsigned int i;
   fprintf(stderr, "node %d submit slave %s wd %d with %d deps, copies are: ", gasnet_mynode(), /*(((WG*) wd)->getParent() == (WG*) GASNetAPI::_rwgGPU ? "GPU" : "SMP")*/"n/a", wd->getId(), deps );
   for ( i = 0; i < wd->getNumCopies(); i++)
      fprintf(stderr, "%s%s:%p ", ( wd->getCopies()[i].isInput() ? "r" : "-" ), ( wd->getCopies()[i].isOutput() ? "w" : "-" ), (void *) wd->getCopies()[i].getAddress() );
   fprintf(stderr, "\n");
#endif
}

void GASNetAPI::enqueuePutReq( unsigned int dest, void *origAddr, void *destAddr, std::size_t len, void *tmpBuffer )
{
   struct putReqDesc *prd = NEW struct putReqDesc();
   prd->dest = dest;
   prd->origAddr = origAddr;
   prd->destAddr = destAddr;
   prd->len = len;
   prd->tmpBuffer = tmpBuffer;

   _putReqsLock.acquire();
   _putReqs.push_back( prd );
   _putReqsLock.release();
}

void GASNetAPI::checkForPutReqs()
{
   struct putReqDesc *prd = NULL;
   if ( _putReqsLock.tryAcquire() )
   {
      if ( !_putReqs.empty() )
      {
         prd = _putReqs.front();
         _putReqs.pop_front();
      }
      _putReqsLock.release();
   }
   if ( prd != NULL )
   {
      NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-in") );
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER_IN, key, (nanos_event_value_t) prd->len) );
      _put( prd->dest, (uint64_t) prd->destAddr, prd->origAddr, prd->len, prd->tmpBuffer );
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
      delete prd;
   }
}

void GASNetAPI::releaseWDsFromDataDep( void *addr )
{
   _depsLock.acquire();
   //if (gasnet_mynode() == 3)fprintf(stderr, "free dep for addr %p\n", addr );
   std::multimap<uint64_t, wdDeps *>::iterator depIt;
   std::pair <std::multimap<uint64_t, wdDeps *>::iterator, std::multimap<uint64_t, wdDeps *>::iterator> allWdsWithDeps = _depsMap.equal_range( (uint64_t) addr );
   if ( allWdsWithDeps.first != allWdsWithDeps.second )
   {
      for ( depIt = allWdsWithDeps.first; depIt != allWdsWithDeps.second; ++depIt )
      {
         (depIt->second)->count -= 1;
         if ( (depIt->second)->count == 0 ) 
         {
#ifdef HALF_PRESEND
            if ( wdindc++ == 0 ) { sys.submit( *(depIt->second)->wd ); /*std::cerr<<"n:" <<gasnet_mynode()<< " submitted(2) wd " << ((depIt->second)->wd)->getId() <<std::endl; */}
            else {  buffWD = (depIt->second)->wd ; /*std::cerr<<"n:" <<gasnet_mynode()<< " saved wd(2) " << buffWD->getId() <<std::endl; */}
#else
//message("submit wd " << (depIt->second)->wd->getId() );
         //if (gasnet_mynode() == 3) print_copies( (depIt->second)->wd, -1 );
            sys.submit( *(depIt->second)->wd );
#endif
            delete (depIt->second);

         }
      }
      _depsMap.erase( (uint64_t) addr );
   }
   else
   {
      _recvdDeps.insert( (uint64_t) addr ); 
   }
   _depsLock.release();
}

void GASNetAPI::testForDependencies(WD* localWD, std::vector<uint64_t> *deps)
{
   unsigned int i, j, numCopies = localWD->getNumCopies(), numDeps;
   CopyData *newCopies = localWD->getCopies();
   numDeps = (deps == NULL ? 0 : deps->size() );
   if ( numDeps > 0 )
   {
      wdDeps *thisWdDeps = NEW wdDeps;
      thisWdDeps->count = 0;
      thisWdDeps->wd = localWD;
      _depsLock.acquire();
      for (i = 0; i < numDeps; i++)
      {
         std::set<uint64_t>::iterator recvDepsIt = _recvdDeps.find( (*deps)[i] );
         if ( recvDepsIt == _recvdDeps.end() ) 
         {
            thisWdDeps->count += 1;
            //if (gasnet_mynode() == 3)fprintf(stderr, "ADD TO DEPS node %d: wd id %d dep (by deps) with tag %p\n", gasnet_mynode(), localWD->getId(), (void *)newCopies[i].getAddress() );
            _depsMap.insert( std::pair<uint64_t, wdDeps*> ( (*deps)[i], thisWdDeps ) ); 
         }
         else
         {
            _recvdDeps.erase( recvDepsIt );
         }
      }
      for (i = 0; i < numCopies; i += 1) {
         bool already_in_deps = false;
         for ( j = 0; j < numDeps && !already_in_deps ; j += 1 )
         {
            already_in_deps = ( (*deps)[j] == newCopies[i].getAddress() );
            //if (gasnet_mynode() == 3)fprintf(stderr, "node %d: wd id %d dep is addr %p vs dep %p\n", gasnet_mynode(), localWD->getId(), (void*)newCopies[i].getAddress(), (void*)(*deps)[j] );
         }
         //if (gasnet_mynode() == 3)fprintf(stderr, "node %d: wd id %d dep is already in deps? %p : %s\n", gasnet_mynode(), localWD->getId(), (void *)newCopies[i].getAddress(), (char *)( already_in_deps ? "yes" : "no") );
         if ( !already_in_deps )
         {
            if ( _depsMap.find( newCopies[i].getAddress() ) != _depsMap.end() ) 
            {
               thisWdDeps->count += 1;
               //if (gasnet_mynode() == 3)fprintf(stderr, "node %d: wd id %d dep (by copies) with tag %p\n", gasnet_mynode(), localWD->getId(), (void *)newCopies[i].getAddress() );
               _depsMap.insert( std::pair<uint64_t, wdDeps*> ( newCopies[i].getAddress(), thisWdDeps ) ); 
            }
         }
      }
      _depsLock.release();
      if ( thisWdDeps->count == 0) 
      {
#ifdef HALF_PRESEND
         if ( wdindc++ == 0 ) { sys.submit( *localWD ); /* std::cerr<<"n:" <<gasnet_mynode()<< " submitted wd " << localWD->getId() <<std::endl; */}
         else { buffWD = localWD; /* std::cerr<<"n:" <<gasnet_mynode()<< " saved wd " << buffWD->getId() <<std::endl; */}
#else
         //message("submit wd " << localWD->getId() );
         //if (gasnet_mynode() == 3)print_copies( localWD, numDeps );
         sys.submit( *localWD );
#endif
         delete thisWdDeps;
      }
   }
   else 
   {
      wdDeps *thisWdDeps = NULL;
      _depsLock.acquire();
      for (i = 0; i < numCopies; i += 1) {
         if ( _depsMap.find( newCopies[i].getAddress() ) != _depsMap.end() ) 
         {
            if ( thisWdDeps == NULL ) {
               thisWdDeps = NEW wdDeps;
               thisWdDeps->count = 1;
               thisWdDeps->wd = localWD;
            }
            else thisWdDeps->count += 1;

            //if (gasnet_mynode() == 3)fprintf(stderr, "node %d: wd id %d dep (by copies) with tag %p\n", gasnet_mynode(), localWD->getId(), (void *)newCopies[i].getAddress() );
            _depsMap.insert( std::pair<uint64_t, wdDeps*> ( newCopies[i].getAddress(), thisWdDeps ) ); 
         }
      }
      _depsLock.release();
      if ( thisWdDeps == NULL)
      {
#ifdef HALF_PRESEND
         if ( wdindc++ == 0 ) { sys.submit( *localWD ); /* std::cerr<<"n:" <<gasnet_mynode()<< " submitted+ wd " << localWD->getId() <<std::endl;*/ }
         else { buffWD = localWD; /*std::cerr<<"n:" <<gasnet_mynode()<< " saved+ wd " << buffWD->getId() <<std::endl;*/ }
#else
         //message("submit wd " << localWD->getId() );
         //if (gasnet_mynode() == 3)print_copies( localWD, numDeps );
         sys.submit( *localWD );
#endif
      }
   }
}

void GASNetAPI::enqueueFreeBufferNotify( void *tmpBuffer )
{
   _freeBufferReqsLock.acquire();
   _freeBufferReqs.push_back( tmpBuffer );
   _freeBufferReqsLock.release();
}

void GASNetAPI::checkForFreeBufferReqs()
{
   void *addr = NULL;
   if ( _freeBufferReqsLock.tryAcquire() )
   {
   //message("process free reqs");
      if ( !_freeBufferReqs.empty() )
      {
         addr = _freeBufferReqs.front();
         _freeBufferReqs.pop_front();
      }
      _freeBufferReqsLock.release();
   //message("end of process free reqs");
   }
   if ( addr != NULL )
      sendFreeTmpBuffer( addr );
}

void GASNetAPI::checkWorkDoneReqs()
{
   std::pair<void *, unsigned int> rwd(NULL, 0);
   if ( _workDoneReqsLock.tryAcquire() )
   {
   //message("process free reqs");
      if ( !_workDoneReqs.empty() )
      {
         rwd = _workDoneReqs.front();
         _workDoneReqs.pop_front();
      }
      _workDoneReqsLock.release();
   //message("end of process free reqs");
   }
   if ( rwd.first != NULL )
      _sendWorkDoneMsg( 0, rwd.first, rwd.second );
}
void GASNetAPI::checkDeferredWorkReqs()
{
   std::pair<unsigned int, std::pair< WD*, std::vector<uint64_t> *> > dwd ( 0, std::pair< WD*, std::vector<uint64_t> *> ( NULL, NULL )  );
   if ( _deferredWorkReqsLock.tryAcquire() )
   {
      //message("process free reqs");
      if ( !_deferredWorkReqs.empty() )
      {
         dwd = _deferredWorkReqs.front();
         _deferredWorkReqs.pop_front();
      }
      if ( dwd.second.first != NULL )
      {
         if (dwd.first == _recvSeqN.value() ) 
         {
            _deferredWorkReqsLock.release();
            _recvSeqN++;
            testForDependencies( dwd.second.first, dwd.second.second );
            checkDeferredWorkReqs();
         }
         else {
            _deferredWorkReqs.push_front( dwd );
            _deferredWorkReqsLock.release();
         }
      }
      else {
         _deferredWorkReqsLock.release();
      }
      //message("end of process free reqs");
   }
}

void GASNetAPI::amFinalize(gasnet_token_t token)
{
   gasnet_node_t src_node;
   if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error obtaining node information.\n");
   }
   //gasnet_AMReplyShort0(token, 204);
   sys.stopFirstThread();
}

void GASNetAPI::amFinalizeReply(gasnet_token_t token)
{
   gasnet_node_t src_node;
   if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error obtaining node information.\n");
   }
}

void GASNetAPI::amWork(gasnet_token_t token, void *arg, std::size_t argSize,
      gasnet_handlerarg_t workLo,
      gasnet_handlerarg_t workHi,
      gasnet_handlerarg_t xlateLo,
      gasnet_handlerarg_t xlateHi,
      gasnet_handlerarg_t rmwdLo,
      gasnet_handlerarg_t rmwdHi,
      unsigned int dataSize, unsigned int wdId, unsigned int numPe, int arch, unsigned int seq )
{
   void (*work)( void *) = (void (*)(void *)) MERGE_ARG( workHi, workLo );
   void (*xlate)( void *, void *) = (void (*)(void *, void *)) MERGE_ARG( xlateHi, xlateLo );
   void *rmwd = (void *) MERGE_ARG( rmwdHi, rmwdLo );
   gasnet_node_t src_node;
   unsigned int i;
   WorkGroup *rwg;

   if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error obtaining node information.\n");
   }
   {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
         NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( wdId ) ; )
         NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_AM_WORK, id, 0, 0, src_node ); )
   }
   char *work_data = NULL;
   std::size_t work_data_len = 0;

   if ( work_data == NULL )
   {
      work_data = NEW char[ argSize ];
      memcpy( work_data, arg, argSize );
   }
   else
   {
      fatal0("Unsupported: work_data bigger than a max gasnet request.");
      memcpy( &work_data[ work_data_len ], arg, argSize );
   }

   nanos_smp_args_t smp_args;
   smp_args.outline = (void (*)(void *)) work;

   WD *localWD = NULL;
   char *data = NULL;
   unsigned int numCopies = *((int *) &work_data[ dataSize ]);
   CopyData *newCopies = NULL;
   CopyData **newCopiesPtr = ( numCopies > 0 ) ? &newCopies : NULL ;
   nanos_device_t newDeviceSMP = { local_nanos_smp_factory, sizeof(SMPDD), (void *) &smp_args } ;
#ifdef GPU_DEV
   nanos_device_t newDeviceGPU = { local_nanos_gpu_factory, sizeof(GPUDD), (void *) &smp_args } ;
#endif
   nanos_device_t *devPtr = NULL;

   if (arch == 0)
   {
      //SMP
      devPtr = &newDeviceSMP;

      if (GASNetAPI::_rwgSMP == NULL) 
         GASNetAPI::_rwgSMP = ClusterInfo::getRemoteWorkGroup( 0 );

      rwg = (WorkGroup *) GASNetAPI::_rwgSMP;
   }
#ifdef GPU_DEV
   else if (arch == 1)
   {
      //FIXME: GPU support
      devPtr = &newDeviceGPU;

      if (GASNetAPI::_rwgGPU == NULL)
         GASNetAPI::_rwgGPU = ClusterInfo::getRemoteWorkGroup( 1 );

      rwg = (WorkGroup *) GASNetAPI::_rwgGPU;
   }
#endif
   else
   {
      rwg = NULL;
      fprintf(stderr, "Unsupported architecture\n");
   }

   //if (gasnet_mynode() == 3) { message("n:3 amWork id " << wdId << " seq is " << seq << " recvd seq counter is " << _recvSeqN.value() ); }
   sys.createWD( &localWD, (std::size_t) 1, devPtr, (std::size_t) dataSize, (int) ( sizeof(void *) ), (void **) &data, (WG *)rwg, (nanos_wd_props_t *) NULL, (std::size_t) numCopies, newCopiesPtr, xlate, 0, NULL );

   std::memcpy(data, work_data, dataSize);

   unsigned int numDeps = *( ( int * ) &work_data[ dataSize + sizeof( int ) + numCopies * sizeof( CopyData ) ] );
   uint64_t *depTags = ( ( uint64_t * ) &work_data[ dataSize + sizeof( int ) + numCopies * sizeof( CopyData ) +sizeof( int ) ] );

   for (i = 0; i < numCopies; i += 1)
      new ( &newCopies[i] ) CopyData( *( ( CopyData *) &work_data[ dataSize + sizeof( int ) + i * sizeof( CopyData ) ] ) );

   localWD->setId( wdId );
   localWD->setRemoteAddr( rmwd );
   {
           NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
           NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) wdId)  )  ; )
           NANOS_INSTRUMENT ( instr->createDeferredPtPEnd ( *localWD, NANOS_WD_REMOTE, id, 0, NULL, NULL, 0 ); )
   }

   std::vector<uint64_t> *depsv = NULL;
   if ( numDeps > 0)
   {
      depsv = NEW std::vector<uint64_t>( numDeps );
      for (i = 0; i < numDeps; i++)
         (*depsv)[i] = depTags[i];
   }

   if ( _recvSeqN.value() == seq )
   {
    _recvSeqN++;
    testForDependencies( localWD, depsv );
    checkDeferredWorkReqs();
   
#if 0
   if ( numDeps > 0 )
   {
      wdDeps *thisWdDeps = NEW wdDeps;
      thisWdDeps->count = 0;
      thisWdDeps->wd = localWD;
      _depsLock.acquire();
      for (i = 0; i < numDeps; i++)
      {
         std::set<uint64_t>::iterator recvDepsIt = _recvdDeps.find( depTags[i] );
         if ( recvDepsIt == _recvdDeps.end() ) 
         {
            thisWdDeps->count += 1;
            if (gasnet_mynode() == 3)fprintf(stderr, "ADD TO DEPS node %d: wd id %d dep (by deps) with tag %p\n", gasnet_mynode(), wdId, (void *)newCopies[i].getAddress() );
            _depsMap.insert( std::pair<uint64_t, wdDeps*> ( depTags[i], thisWdDeps ) ); 
         }
         else
         {
            _recvdDeps.erase( recvDepsIt );
         }
      }
      for (i = 0; i < numCopies; i += 1) {
         bool already_in_deps = false;
         for ( j = 0; j < numDeps && !already_in_deps ; j += 1 )
         {
            already_in_deps = ( depTags[j] == newCopies[i].getAddress() );
         if (gasnet_mynode() == 3)fprintf(stderr, "node %d: wd id %d dep is addr %p vs dep %p\n", gasnet_mynode(), wdId, (void*)newCopies[i].getAddress(), (void*)depTags[j] );
         }
         if (gasnet_mynode() == 3)fprintf(stderr, "node %d: wd id %d dep is already in deps? %p : %s\n", gasnet_mynode(), wdId, (void *)newCopies[i].getAddress(), (char *)( already_in_deps ? "yes" : "no") );
         if ( !already_in_deps )
         {
            if ( _depsMap.find( newCopies[i].getAddress() ) != _depsMap.end() ) 
            {
               thisWdDeps->count += 1;
               if (gasnet_mynode() == 3)fprintf(stderr, "node %d: wd id %d dep (by copies) with tag %p\n", gasnet_mynode(), wdId, (void *)newCopies[i].getAddress() );
               _depsMap.insert( std::pair<uint64_t, wdDeps*> ( newCopies[i].getAddress(), thisWdDeps ) ); 
            }
         }
      }
      _depsLock.release();
      if ( thisWdDeps->count == 0) 
      {
#ifdef HALF_PRESEND
         if ( wdindc++ == 0 ) { sys.submit( *localWD ); /* std::cerr<<"n:" <<gasnet_mynode()<< " submitted wd " << localWD->getId() <<std::endl; */}
         else { buffWD = localWD; /* std::cerr<<"n:" <<gasnet_mynode()<< " saved wd " << buffWD->getId() <<std::endl; */}
#else
//message("submit wd " << localWD->getId() );
         if (gasnet_mynode() == 3)print_copies( localWD, numDeps );
         sys.submit( *localWD );
#endif
         delete thisWdDeps;
      }
   }
   else 
   {
      wdDeps *thisWdDeps = NULL;
      _depsLock.acquire();
      for (i = 0; i < numCopies; i += 1) {
         if ( _depsMap.find( newCopies[i].getAddress() ) != _depsMap.end() ) 
         {
            if ( thisWdDeps == NULL ) {
               thisWdDeps = NEW wdDeps;
               thisWdDeps->count = 1;
               thisWdDeps->wd = localWD;
            }
            else thisWdDeps->count += 1;

            if (gasnet_mynode() == 3)fprintf(stderr, "node %d: wd id %d dep (by copies) with tag %p\n", gasnet_mynode(), wdId, (void *)newCopies[i].getAddress() );
            _depsMap.insert( std::pair<uint64_t, wdDeps*> ( newCopies[i].getAddress(), thisWdDeps ) ); 
         }
      }
      _depsLock.release();
      if ( thisWdDeps == NULL)
      {
#ifdef HALF_PRESEND
         if ( wdindc++ == 0 ) { sys.submit( *localWD ); /* std::cerr<<"n:" <<gasnet_mynode()<< " submitted+ wd " << localWD->getId() <<std::endl;*/ }
         else { buffWD = localWD; /*std::cerr<<"n:" <<gasnet_mynode()<< " saved+ wd " << buffWD->getId() <<std::endl;*/ }
#else
//message("submit wd " << localWD->getId() );
         if (gasnet_mynode() == 3)print_copies( localWD, numDeps );
         sys.submit( *localWD );
#endif
      }
   }
#endif
   } else { //not expected seq number, enqueue
      _deferredWorkReqsLock.acquire();
      _deferredWorkReqs.push_back( std::pair< unsigned int, std::pair < WD*, std::vector<uint64_t> *> >( seq, std::pair< WD*, std::vector<uint64_t> * > ( localWD, depsv ) ) );
      _deferredWorkReqsLock.release();
   }

   delete work_data;
   work_data = NULL;
   work_data_len = 0;
}

void GASNetAPI::amWorkData(gasnet_token_t token, void *buff, std::size_t len,
      gasnet_handlerarg_t msgNum,
      gasnet_handlerarg_t totalLenLo,
      gasnet_handlerarg_t totalLenHi)
{
   gasnet_node_t src_node;
   //std::size_t totalLen = (std::size_t) MERGE_ARG( totalLenHi, totalLenLo );
   if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error obtaining node information.\n");
   }

   std::cerr<<"UNSUPPORTED FOR NOW"<<std::endl;
}

void GASNetAPI::amWorkDone( gasnet_token_t token, gasnet_handlerarg_t addrLo, gasnet_handlerarg_t addrHi, gasnet_handlerarg_t peId )
{
   gasnet_node_t src_node;
   void * addr = (void *) MERGE_ARG( addrHi, addrLo );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( addr ) ; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_AM_WORK_DONE, id, 0, 0, src_node ); )

   //message("wd completed " << ((WD*)addr)->getId() );
   sys.getNetwork()->notifyWorkDone( src_node, addr, peId );
}

void GASNetAPI::amMalloc( gasnet_token_t token, gasnet_handlerarg_t sizeLo, gasnet_handlerarg_t sizeHi,
      gasnet_handlerarg_t waitObjAddrLo, gasnet_handlerarg_t waitObjAddrHi )
{
   gasnet_node_t src_node;
   void *addr = NULL; volatile int *ptr;
   std::size_t size = ( std::size_t ) MERGE_ARG( sizeHi, sizeLo );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
//	#if GPU_DEV
//	addr = GPUDevice::allocatePinnedMemory2( size );
//	#else
   addr = std::malloc( ( std::size_t ) size );
   
   ptr = (volatile int *)addr;
  
   for (unsigned int i = 0; i < ( size / sizeof(int) ); i+= 1024 ) { ptr[i] = ptr[i]; }
	//#endif
   //std::cerr<<"Malloc size " << size << " returns "<<addr<<std::endl;
   if ( addr == NULL )
   {
      message0 ( "I could not allocate " << (std::size_t) size << " (sizeof std::size_t is " << sizeof(std::size_t) << " ) " << (void *) size << " bytes of memory on node " << gasnet_mynode() << ". Try setting NX_CLUSTER_NODE_MEMORY to a lower value." );
      fatal0 ("I can not continue." );
   }
   if ( gasnet_AMReplyShort4( token, 208, ( gasnet_handlerarg_t ) ARG_LO( addr ),
            ( gasnet_handlerarg_t ) ARG_HI( addr ),
            ( gasnet_handlerarg_t ) waitObjAddrLo,
            ( gasnet_handlerarg_t ) waitObjAddrHi ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error sending a message to node %d.\n", src_node );
   }
}

void GASNetAPI::amMallocReply( gasnet_token_t token, gasnet_handlerarg_t addrLo, gasnet_handlerarg_t addrHi,
      gasnet_handlerarg_t waitObjAddrLo, gasnet_handlerarg_t waitObjAddrHi )
{
   void * addr = ( void * ) MERGE_ARG( addrHi, addrLo );
   Network::mallocWaitObj *request = ( Network::mallocWaitObj * ) MERGE_ARG( waitObjAddrHi, waitObjAddrLo );
   gasnet_node_t src_node;
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   sys.getNetwork()->notifyMalloc( src_node, addr, request );
}

void GASNetAPI::amFree( gasnet_token_t token, gasnet_handlerarg_t addrLo, gasnet_handlerarg_t addrHi )
{
   void * addr = (void *) MERGE_ARG( addrHi, addrLo );
   gasnet_node_t src_node;

   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   free( addr );
}

void GASNetAPI::amRealloc( gasnet_token_t token, gasnet_handlerarg_t oldAddrLo, gasnet_handlerarg_t oldAddrHi,
      gasnet_handlerarg_t oldSizeLo, gasnet_handlerarg_t oldSizeHi,
      gasnet_handlerarg_t newAddrLo, gasnet_handlerarg_t newAddrHi,
      gasnet_handlerarg_t newSizeLo, gasnet_handlerarg_t newSizeHi)
{
   void * oldAddr = (void *) MERGE_ARG( oldAddrHi, oldAddrLo );
   void * newAddr = (void *) MERGE_ARG( newAddrHi, newAddrLo );
   std::size_t oldSize = (std::size_t) MERGE_ARG( oldSizeHi, oldSizeLo );
   //std::size_t newSize = (std::size_t) MERGE_ARG( newSizeHi, newSizeLo );
   gasnet_node_t src_node;

   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) newAddr );
   if (ent != NULL) 
   { 
      if (ent->getOwner() != NULL) 
      {
         ent->getOwner()->discard( *_masterDir, (uint64_t) newAddr, ent);
         std::cerr << "REALLOC WARNING, newAddr had an entry n:" << gasnet_mynode() << " discarding tag " << (void *) newAddr << std::endl;
      }
   }
   std::memcpy( newAddr, oldAddr, oldSize );
}

void GASNetAPI::amMasterHostname( gasnet_token_t token, void *buff, std::size_t nbytes )
{
   gasnet_node_t src_node;
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   /* for now we only allow this at node 0 */
   if ( src_node == 0 )
   {
      sys.getNetwork()->setMasterHostname( ( char  *) buff );
   }
}

void GASNetAPI::amPut( gasnet_token_t token,
      void *buf,
      std::size_t len,
      gasnet_handlerarg_t realAddrLo,
      gasnet_handlerarg_t realAddrHi,
      gasnet_handlerarg_t realTagLo,
      gasnet_handlerarg_t realTagHi,
      gasnet_handlerarg_t firstMsg,
      gasnet_handlerarg_t lastMsg )
{
   gasnet_node_t src_node;
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   void *realAddr = ( int * ) MERGE_ARG( realAddrHi, realAddrLo );
   void *realTag = ( int * ) MERGE_ARG( realTagHi, realTagLo );

   rxBytes += len;

   _waitingPutRequestsLock.acquire();
   std::set<void *>::iterator it;
   if ( ( it = _waitingPutRequests.find( realTag ) ) != _waitingPutRequests.end() ) //we have to wait 
   {
      //delayEnque( src_node, origAddr, destAddr, tagAddr, waitObj, realLen );
      //message("addr " << buf << " found! erasing from waiting list");
      _waitingPutRequests.erase( it );
   }
   else
   {
      _receivedUnmatchedPutRequests.insert( realTag );
   }
   _waitingPutRequestsLock.release();
   
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = len; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( buf ) ; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_PUT, id, sizeKey, xferSize, src_node ); )

   if ( firstMsg )
   {
   //if (src_node != 0 ) fprintf(stderr, "BEGIN amPut to node %d, from %d, local addr %p, realTag %p data is %f\n", gasnet_mynode(), src_node, buf, realTag, *((float *)buf));
      DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) realTag );
      if (ent != NULL) 
      { 
         if (ent->getOwner() != NULL) {
            ent->getOwner()->discard( *_masterDir, (uint64_t) realTag, ent);
         } else {
            ent->increaseVersion();
         }
      }
   }
   if ( realAddr != NULL )
   {
      //message0("memcopy in node, local tmp buffer " << (void*) buf);
      ::memcpy( realAddr, buf, len );
   }
   if ( lastMsg )
   {
      //send notify to node 
      // compute the original local address
      uintptr_t localAddr =  ( ( uintptr_t ) buf ) - ( ( uintptr_t ) realAddr - ( uintptr_t ) realTag );
      //if ( firstMsg ) { if ((char *) localAddr != buf) fprintf(stderr, "SOMETHING WRONG HERE, realAddr is %p, realTag is %p, but localAddr is %p\n", realAddr, realTag, (void *) localAddr ); }
      enqueueFreeBufferNotify( ( void * ) localAddr );
      if ( src_node > 0 )
      {
         releaseWDsFromDataDep( realTag );
      }
   //if (src_node != 0 )fprintf(stderr, "END amPut to node %d, from %d, local addr %p, realTag %p, data is %f\n", gasnet_mynode(), src_node, buf, realTag, *((float *)buf));
   }
}

void GASNetAPI::amGetReply( gasnet_token_t token,
      void *buf,
      std::size_t len,
      gasnet_handlerarg_t waitObjLo,
      gasnet_handlerarg_t waitObjHi)
{
   gasnet_node_t src_node;
   int *waitObj = ( int * ) MERGE_ARG( waitObjHi, waitObjLo );

   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   //fprintf(stderr, "get reply from %d: data is %f\n", src_node , *((float *)buf));
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = len; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) waitObj; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent ( NANOS_XFER_GET, id, sizeKey, xferSize, src_node ); )

   if ( waitObj != NULL )
   {
      *waitObj = 1;
   }
}

void GASNetAPI::amGet( gasnet_token_t token,
      gasnet_handlerarg_t destAddrLo,
      gasnet_handlerarg_t destAddrHi,
      gasnet_handlerarg_t origAddrLo,
      gasnet_handlerarg_t origAddrHi,
      gasnet_handlerarg_t tagAddrLo,
      gasnet_handlerarg_t tagAddrHi,
      gasnet_handlerarg_t lenLo,
      gasnet_handlerarg_t lenHi,
      gasnet_handlerarg_t waitObjLo,
      gasnet_handlerarg_t waitObjHi )
{
   gasnet_node_t src_node;
   void *origAddr = ( void * ) MERGE_ARG( origAddrHi, origAddrLo );
   void *destAddr = ( void * ) MERGE_ARG( destAddrHi, destAddrLo );
   void *tagAddr = ( void * ) MERGE_ARG( tagAddrHi, tagAddrLo );
   NANOS_INSTRUMENT ( int *waitObj = ( int * ) MERGE_ARG( waitObjHi, waitObjLo ); )
   std::size_t realLen = ( std::size_t ) MERGE_ARG( lenHi, lenLo );

   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   txBytes += realLen;

   if ( tagAddr == origAddr )
   {
      _waitingPutRequestsLock.acquire();
      if ( _waitingPutRequests.find( tagAddr ) != _waitingPutRequests.end() ) //we have to wait 
      {
         //delayAmGet( src_node, tagAddr, destAddr, tagAddr, waitObj, realLen );
         message("addr " << tagAddr << " found! erasing from waiting list");
         message("WARNING: this amGet SHOULD BE DELAYED!!!");
      } //else { message("addr " << tagAddr << " not found at waiting list");}
      _waitingPutRequestsLock.release();

      DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) tagAddr );
      if (ent != NULL) 
      {
         if (ent->getOwner() != NULL )
            if ( !ent->isInvalidated() )
            {
               std::list<uint64_t> tagsToInvalidate;
               tagsToInvalidate.push_back( ( uint64_t ) tagAddr );
               _masterDir->synchronizeHost( tagsToInvalidate );
            }
      }
   }
   {
      //fprintf(stderr, "n:%d thd %d am_xfer_get: srcAddr=%p, srcHi=%p, srcLo=%p, dstAddr=%p, dstHi=%p, dstLo=%p res=%f, waitObj=%p\n", gasnet_mynode(), myThread->getId(), origAddr, (void *)origAddrHi, (void *)origAddrLo, destAddr, (void*)destAddrHi, (void*)destAddrLo, *((float*)origAddr), /*waitObj*/(void *)NULL );

      //fprintf(stderr, "n:%d thd %d am_xfer_get: srcAddr=%p, tagAddr=%p, destAddr=%p len=%ld res=%f, max=%ld\n", gasnet_mynode(), myThread->getId(), origAddr, tagAddr, destAddr, realLen,  *((float*)origAddr), gasnet_AMMaxLongReply() );
      if ( ( unsigned int ) realLen <= gasnet_AMMaxLongReply() )
      {
         NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
         NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
         NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
         NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
         NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) waitObj; )
         NANOS_INSTRUMENT ( instr->raiseClosePtPEvent ( NANOS_XFER_GET, id, sizeKey, xferSize, src_node ); )

         NANOS_INSTRUMENT ( xferSize = realLen; )
         NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_GET, id, sizeKey, xferSize, src_node ); )

         if ( gasnet_AMReplyLong2( token, 212, origAddr, realLen, destAddr, waitObjLo, waitObjHi ) != GASNET_OK )
         {
            fprintf( stderr, "gasnet: Error sending reply msg.\n" );
         }
         //fprintf(stderr, "n:%d thd %d am_xfer_get: srcAddr=%p, srcHi=%p, srcLo=%p, dstAddr=%p, dstHi=%p, dstLo=%p res=%f, size=%ld\n", gasnet_mynode(), myThread->getId(), origAddr, (void *)origAddrHi, (void *)origAddrLo, destAddr, (void*)destAddrHi, (void*)destAddrLo, *((float*)origAddr), realLen );

      }
      else
      {
         fprintf( stderr, "gasnet: Error, requested a GET of size > gasnet_AMMaxLongReply() bytes.\n" );
      }
   }
}

void GASNetAPI::amPutF( gasnet_token_t token,
      gasnet_handlerarg_t destAddrLo,
      gasnet_handlerarg_t destAddrHi,
      gasnet_handlerarg_t len,
      gasnet_handlerarg_t wordSize,
      gasnet_handlerarg_t valueLo,
      gasnet_handlerarg_t valueHi )
{
   gasnet_node_t src_node;
   int i;
   void *destAddr = ( void * ) MERGE_ARG( destAddrHi, destAddrLo );
   uint64_t value = ( uint64_t ) MERGE_ARG( valueHi, valueLo );

   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   uint64_t *ptr64 = ( uint64_t * ) destAddr;
   uint32_t *ptr32 = ( uint32_t * ) destAddr;
   uint16_t *ptr16 = ( uint16_t * ) destAddr;
   uint8_t *ptr8  = ( uint8_t * ) destAddr;

   uint64_t val64 = ( uint64_t ) value;
   uint32_t val32 = ( uint32_t ) value;
   uint16_t val16 = ( uint16_t ) value;
   uint8_t val8  = ( uint8_t ) value;

   switch ( wordSize )
   {
      case 8:
         for ( i = 0; i < (len/8) ; i++ )
         {
            ptr64[ i ] = val64;
         }
         break;
      case 4:
         for ( i = 0; i < (len/4) ; i++ )
         {
            ptr32[ i ] = val32;
         }
         break;
      case 2:
         for ( i = 0; i < (len/2) ; i++ )
         {
            ptr16[ i ] = val16;
         }
         break;
      case 1:
         for ( i = 0; i < len ; i++ )
         {
            ptr8[ i ] = val8;
         }
         break;
      default:
         break;
   }

   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = len; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( destAddr ) ; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_PUT, id, sizeKey, xferSize, src_node ); )
}

void GASNetAPI::amRequestPut( gasnet_token_t token,
      gasnet_handlerarg_t destAddrLo,
      gasnet_handlerarg_t destAddrHi,
      gasnet_handlerarg_t origAddrLo,
      gasnet_handlerarg_t origAddrHi,
      gasnet_handlerarg_t tmpBufferLo,
      gasnet_handlerarg_t tmpBufferHi,
      gasnet_handlerarg_t lenLo,
      gasnet_handlerarg_t lenHi,
      gasnet_handlerarg_t dst )
{
   gasnet_node_t src_node;
   void *origAddr = ( void * ) MERGE_ARG( origAddrHi, origAddrLo );
   void *destAddr = ( void * ) MERGE_ARG( destAddrHi, destAddrLo );
   void *tmpBuffer = ( void * ) MERGE_ARG( tmpBufferHi, tmpBufferLo );
   std::size_t len = ( std::size_t ) MERGE_ARG( lenHi, lenLo );

   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   _waitingPutRequestsLock.acquire();
   if ( _waitingPutRequests.find( origAddr ) != _waitingPutRequests.end() ) //we have to wait 
   {
      //delayEnque( src_node, origAddr, destAddr, tagAddr, waitObj, realLen );
      message("WARNING: this amRequestPut SHOULD BE DELAYED!!!");
   }
   _waitingPutRequestsLock.release();

   enqueuePutReq( dst, origAddr, destAddr, len, tmpBuffer );
}

void GASNetAPI::amWaitRequestPut( gasnet_token_t token, 
      gasnet_handlerarg_t addrLo,
      gasnet_handlerarg_t addrHi )
{
   void *addr = ( void * ) MERGE_ARG( addrHi, addrLo );
   std::set< void * >::iterator it;

   _waitingPutRequestsLock.acquire();
	//message("processing sendWaitForRequestPut( this is node " << gasnet_mynode() << ", addr "<< (void *) addr <<").");
   if ( _receivedUnmatchedPutRequests.empty() || ( it = _receivedUnmatchedPutRequests.find( addr ) ) == _receivedUnmatchedPutRequests.end() )
   {
      // no unmatched Put Request, so add it into the waiting set
      _waitingPutRequests.insert( addr );
      //message("addr " << addr << " added to waiting list");
   } 
   else
   {
      // unmatched Put Request found, it looks like we already received it so just remove it 
      _receivedUnmatchedPutRequests.erase( it );
   }
   _waitingPutRequestsLock.release();
	//message("processing done");
}

void GASNetAPI::amFreeTmpBuffer( gasnet_token_t token,
      gasnet_handlerarg_t addrLo,
      gasnet_handlerarg_t addrHi )
{
   void *addr = ( void * ) MERGE_ARG( addrHi, addrLo );
   gasnet_node_t src_node;
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   _pinnedAllocatorsLocks[ src_node ]->acquire();
   _pinnedAllocators[ src_node ]->free( addr );
   _pinnedAllocatorsLocks[ src_node ]->release();
}

void GASNetAPI::initialize ( Network *net )
{
   int my_argc = OS::getArgc();
   char **my_argv = OS::getArgv();
   uintptr_t segSize;


   _net = net;

   gasnet_handlerentry_t htable[] = {
      { 203, (void (*)()) amFinalize },
      { 204, (void (*)()) amFinalizeReply },
      { 205, (void (*)()) amWork },
      { 206, (void (*)()) amWorkDone },
      { 207, (void (*)()) amMalloc },
      { 208, (void (*)()) amMallocReply },
      { 209, (void (*)()) amMasterHostname },
      { 210, (void (*)()) amPut },
      { 211, (void (*)()) amGet },
      { 212, (void (*)()) amGetReply },
      { 213, (void (*)()) amPutF },
      { 214, (void (*)()) amRequestPut },
      { 215, (void (*)()) amWorkData },
      { 216, (void (*)()) amFree },
      { 217, (void (*)()) amRealloc },
      { 218, (void (*)()) amWaitRequestPut },
      { 219, (void (*)()) amFreeTmpBuffer }
   };

   gasnet_init( &my_argc, &my_argv );

   segSize = gasnet_getMaxLocalSegmentSize();

   gasnet_attach( htable, sizeof( htable ) / sizeof( gasnet_handlerentry_t ), segSize, 0);

   //if (segSize != (uintptr_t) -1) fprintf(stderr, "gasnet: segment size was %p bytes\n", ( void * ) segSize);
   //#ifndef GASNET_SEGMENT_EVERYTHING
   //else 
   //{
   //   fprintf(stderr, "gasnet error: gasnet segment size is -1.\n"); 
   //   exit(-1);
   //}
   //#endif

   _net->setNumNodes( gasnet_nodes() );
   _net->setNodeNum( gasnet_mynode() );

   _sentData.reserve( _net->getNumNodes() );

   for (unsigned int i = 0; i < _net->getNumNodes(); i++ )
   {
      //_getRequests.push_back( NEW GetRequestCtl );
      _sentData.push_back( NEW std::set<uint64_t> );
   }

   nodeBarrier();

   {
      unsigned int i;
      char myHostname[256];
      if ( gethostname( myHostname, 256 ) != 0 )
      {
         fprintf(stderr, "os: Error getting the hostname.\n");
      }
	//message0("Node " << _net->getNodeNum() << " running " << myHostname );

      if ( _net->getNodeNum() == 0)
      {
	      sys.getNetwork()->setMasterHostname( (char *) myHostname );

	      for ( i = 1; i < _net->getNumNodes() ; i++ )
	      {
		      sendMyHostName( i );
	      }
      }
   }

   nodeBarrier();

#ifndef GASNET_SEGMENT_EVERYTHING
   if ( _net->getNodeNum() == 0)
   {
      if (0) /* low mem mode */
      {
         unsigned int idx;
         fprintf(stderr, "GASNet was NOT configured with GASNET_SEGMENT_EVERYTHING\n");

         gasnet_seginfo_t seginfoTable[ gasnet_nodes() ];
         gasnet_getSegmentInfo( seginfoTable, gasnet_nodes() );

         void *segmentAddr[ gasnet_nodes() ];
         std::size_t segmentLen[ gasnet_nodes() ];

         verbose0( "GasNet segment information:" );
         for ( idx = 0; idx < gasnet_nodes(); idx += 1)
         {
            segmentAddr[ idx ] = seginfoTable[ idx ].addr;
            segmentLen[ idx ] = seginfoTable[ idx ].size;
            verbose0( "\tnode "<< idx << ": @=" << seginfoTable[ idx ].addr << ", len=" << (void *) seginfoTable[ idx ].size );
         }
         ClusterInfo::addSegments( gasnet_nodes(), segmentAddr, segmentLen );
         _thisNodeSegment = NEW SimpleAllocator( ( uintptr_t ) ClusterInfo::getSegmentAddr( 0 ), ClusterInfo::getSegmentLen( 0 ) );
      }
      else
      {
         unsigned int idx;
         unsigned int nodes = gasnet_nodes();
         void *segmentAddr[ nodes ];
         std::size_t segmentLen[ nodes ];
         void *pinnedSegmentAddr[ nodes ];
         std::size_t pinnedSegmentLen[ nodes ];
         
         gasnet_seginfo_t seginfoTable[ nodes ];
         gasnet_getSegmentInfo( seginfoTable, nodes );
         _pinnedAllocators.reserve( nodes );
         _pinnedAllocatorsLocks.reserve( nodes );
         
         _seqN = NEW Atomic<unsigned int>[nodes];
         verbose0( "GasNet segment information:" );
         for ( idx = 0; idx < nodes; idx += 1)
         {
            pinnedSegmentAddr[ idx ] = seginfoTable[ idx ].addr;
            pinnedSegmentLen[ idx ] = seginfoTable[ idx ].size;
	    segmentAddr[ idx ] = _net->malloc( idx, ClusterInfo::getNodeMem() );
	    segmentLen[ idx ] = ClusterInfo::getNodeMem(); 
            verbose0( "\tnode "<< idx << ": @=" << seginfoTable[ idx ].addr << ", len=" << (void *) seginfoTable[ idx ].size );
            _pinnedAllocators[idx] = NEW SimpleAllocator( ( uintptr_t ) pinnedSegmentAddr[ idx ], pinnedSegmentLen[ idx ] );
            _pinnedAllocatorsLocks[idx] =  NEW Lock( );
            new (&_seqN[idx]) Atomic<unsigned int >( 0 );
         }
         _thisNodeSegment = _pinnedAllocators[0];
         ClusterInfo::addSegments( nodes, segmentAddr, segmentLen );
         ClusterInfo::addPinnedSegments( nodes, pinnedSegmentAddr, pinnedSegmentLen );
      }
   }
#else
   if ( _net->getNodeNum() == 0)
   {
      fprintf(stderr, "GASNet was configured with GASNET_SEGMENT_EVERYTHING\n");
      void *segmentAddr[ gasnet_nodes() ];
      std::size_t segmentLen[ gasnet_nodes() ];

      unsigned int idx;
      segmentAddr[ 0 ] = 0; //NEW char [ ClusterInfo::getNodeMem() ];
      segmentLen[ 0 ] = 0; //ClusterInfo::getNodeMem();
      for ( idx = 1; idx < gasnet_nodes(); idx += 1)
      {
         segmentAddr[ idx ] = _net->malloc( idx, ClusterInfo::getNodeMem() );
         segmentLen[ idx ] = ClusterInfo::getNodeMem(); 
      }
      ClusterInfo::addSegments( gasnet_nodes(), segmentAddr, segmentLen );
      //_thisNodeSegment = NEW SimpleAllocator( ( uintptr_t ) ClusterInfo::getSegmentAddr( 0 ), ClusterInfo::getSegmentLen( 0 ) );
   }
   nodeBarrier();
#endif
}

void GASNetAPI::finalize ()
{
   unsigned int i;
   nodeBarrier();
   for ( i = 0; i < _net->getNumNodes(); i += 1 )
   {
      if ( i == _net->getNodeNum() )
      {
         message0( "Node " << _net->getNodeNum() << " stats: Rx: " << rxBytes << " Tx: " << txBytes );
         verbose0( "Node "<< _net->getNodeNum() << " closing the network." );
      }
      nodeBarrier();
   }
   gasnet_exit(0); //Still hanging on MareNostrum
   //exit(0);
}

void GASNetAPI::poll ()
{
   if (myThread != NULL)
   {
      checkForPutReqs();
      checkForFreeBufferReqs();
      checkWorkDoneReqs();
      //checkDeferredWorkReqs();
      gasnet_AMPoll();
   }
   else
      gasnet_AMPoll();
}

void GASNetAPI::sendExitMsg ( unsigned int dest )
{
   if (gasnet_AMRequestShort0( dest, 203 ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GASNetAPI::sendWorkMsg ( unsigned int dest, void ( *work ) ( void * ), unsigned int dataSize, unsigned int wdId, unsigned int numPe, std::size_t argSize, char * arg, void ( *xlate ) ( void *, void * ), int arch, void *remoteWdAddr/*, void *remoteThd*/ )
{
   std::size_t sent = 0;
   unsigned int msgCount = 0;

   int numCopies = *((int *) &arg[ dataSize ]);
   CopyData *copiesForThisWork = (CopyData *) &arg[ dataSize + sizeof(int)];
   int *depCount = ((int *) &arg[ dataSize + sizeof(int) + sizeof(CopyData) * numCopies ]);
   uint64_t *depAddrs = ((uint64_t *) &arg[ dataSize + sizeof(int) + sizeof(CopyData) * numCopies + sizeof(int) ]);

   *depCount = 0;

   for (int i = 0; i < numCopies; i++)
   {
      uint64_t tag = copiesForThisWork[i].getAddress();
      _sentDataLock.acquire();
      std::set<uint64_t >::iterator addrIt = _sentData[ dest ]->find( tag );
      if( addrIt != _sentData[ dest ]->end() )
      {
         //found an element, I had previously sent a PUT REQUEST to reach this node, set up a dependence.
     //if (dest == 3)fprintf(stderr, "added %p to deps for wd %d\n", (void *)tag, wdId);
         depAddrs[ *depCount ] = tag;
         *depCount += 1;
         _sentData[ dest ]->erase( addrIt );
      }
      _sentDataLock.release();
   }

   while ( (argSize - sent) > gasnet_AMMaxMedium() )
   {
      if ( gasnet_AMRequestMedium3( dest, 215, &arg[ sent ], gasnet_AMMaxMedium(),
               msgCount, 
               ARG_LO( argSize ),
               ARG_HI( argSize ) ) != GASNET_OK )
      {
         fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
      }
      msgCount++;
      sent += gasnet_AMMaxMedium();
   }

     //if (dest == 3){message("send work msg to 3 wd "<<  wdId << " seq used is "  <<(_seqN[dest].value()));}
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( wdId ) ; )
   NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_AM_WORK, id, 0, 0, dest ); )

      if (gasnet_AMRequestMedium11( dest, 205, &arg[ sent ], argSize - sent,
               ARG_LO( work ),
               ARG_HI( work ),
               ARG_LO( xlate ),
               ARG_HI( xlate ),
               ARG_LO( remoteWdAddr ),
               ARG_HI( remoteWdAddr ),
               dataSize, wdId, numPe, arch, _seqN[dest]++ ) != GASNET_OK)
      {
         fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
      }
}

void GASNetAPI::sendWorkDoneMsg ( unsigned int dest, void *remoteWdAddr, int peId )
{
   _workDoneReqsLock.acquire();
   _workDoneReqs.push_back( std::pair<void *, unsigned int> ( remoteWdAddr, (unsigned int ) peId) );
   _workDoneReqsLock.release();
}

void GASNetAPI::_sendWorkDoneMsg ( unsigned int dest, void *remoteWdAddr, int peId )
{
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( remoteWdAddr ) ; )
   NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_AM_WORK_DONE, id, 0, 0, dest ); )
#ifdef HALF_PRESEND
   if ( wdindc-- == 2 ) { sys.submit( *buffWD ); /*std::cerr<<"n:" <<gasnet_mynode()<< " submitted wd " << buffWD->getId() <<std::endl;*/} 
#endif
   if (gasnet_AMRequestShort3( dest, 206, 
            ARG_LO( remoteWdAddr ),
            ARG_HI( remoteWdAddr ),
            peId ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GASNetAPI::_put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, std::size_t size, void *remoteTmpBuffer )
{
   std::size_t sent = 0, thisReqSize;
   txBytes += size;
   _totalBytes += size;
#if 0
   unsigned int i = 1;
   unsigned int totalWords;
   unsigned int selectedSize;
   uint64_t value;
   //test the buffer:
   if ( size % 8 == 0 )
   {
      uint64_t *ptr = ( uint64_t * ) localAddr;
      totalWords = size / 8;
      selectedSize = 8;
      value = ptr[ 0 ];
      while ( ptr[ 0 ] == ptr[ i ] )
         i++;
   }
   else if ( size % 4 )
   {
      uint32_t *ptr = (uint32_t *) localAddr;
      totalWords = size / 4;
      selectedSize = 4;
      value = ( uint64_t ) ptr[ 0 ];
      while (ptr[0] == ptr[i])
         i++;
   }
   else if ( size % 2 )
   {
      uint16_t *ptr = (uint16_t *) localAddr;
      totalWords = size / 2;
      selectedSize = 2;
      value = ( uint64_t ) ptr[ 0 ];
      while (ptr[0] == ptr[i])
         i++;
   }
   else
   {
      uint8_t *ptr = (uint8_t *) localAddr;
      totalWords = size;
      selectedSize = 1;
      value = ( uint64_t ) ptr[ 0 ];
      while (ptr[0] == ptr[i])
         i++;
   }
   if ( i == totalWords)
   {
      //fprintf(stderr, "I can do a flash put here!, selected size %d\n", selectedSize);

      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
         NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
         NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
         NANOS_INSTRUMENT ( nanos_event_value_t xferSize = size; )
         NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( remoteAddr ) ; )
         NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_PUT, id, sizeKey, xferSize, remoteNode ); )

         if ( gasnet_AMRequestShort6( remoteNode, 213,
                  ( gasnet_handlerarg_t ) ARG_LO( remoteAddr ),
                  ( gasnet_handlerarg_t ) ARG_HI( remoteAddr ),
                  ( gasnet_handlerarg_t ) size,
                  ( gasnet_handlerarg_t ) selectedSize,
                  ( gasnet_handlerarg_t ) ARG_LO( value ),
                  ( gasnet_handlerarg_t ) ARG_HI( value )) != GASNET_OK )
         {
            fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
         }
   }
   else
#endif
   {
      DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) localAddr );
      if (ent != NULL) 
      {
         if (ent->getOwner() != NULL )
         {
            std::list<uint64_t> tagsToInvalidate;
            tagsToInvalidate.push_back( ( uint64_t ) localAddr );
            _masterDir->synchronizeHost( tagsToInvalidate );
         }
      }
   //fprintf(stderr, "doing put to %d: data is %f\n", remoteNode , *((float *)localAddr ));
      while ( sent < size )
      {
         thisReqSize = ( ( size - sent ) <= gasnet_AMMaxLongRequest() ) ? size - sent : gasnet_AMMaxLongRequest();

         NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
         NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
         NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
         NANOS_INSTRUMENT ( nanos_event_value_t xferSize = thisReqSize; )
         NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( remoteAddr + sent ) ; )
         NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_PUT, id, sizeKey, xferSize, remoteNode ); )

         if ( remoteTmpBuffer != NULL )
         { 
            if ( gasnet_AMRequestLong6( remoteNode, 210,
                     &( ( char *) localAddr )[ sent ],
                     thisReqSize,
                     ( char *) ( ( (char *) remoteTmpBuffer ) + sent ),
                     ARG_LO( ( ( uintptr_t ) ( ( uintptr_t ) remoteAddr ) + sent )),
                     ARG_HI( ( ( uintptr_t ) ( ( uintptr_t ) remoteAddr ) + sent )),
                     ARG_LO( ( ( uintptr_t ) remoteAddr ) ),
                     ARG_HI( ( ( uintptr_t ) remoteAddr ) ),
                     ( sent == 0 ),
                     ( ( sent + thisReqSize ) == size )
                     ) != GASNET_OK)
            {
               fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
            }
         }
         else
         {
            if ( gasnet_AMRequestLong6( remoteNode, 210,
                     &( ( char *) localAddr )[ sent ],
                     thisReqSize,
                     ( char *) ( remoteAddr + sent ),
                     ARG_LO( NULL ),
                     ARG_HI( NULL ),
                     ARG_LO( NULL ),
                     ARG_HI( NULL ),
                     ( sent == 0 ),
                     ( ( sent + thisReqSize ) == size )
                     ) != GASNET_OK)
            {
               fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
            }
         }
         sent += thisReqSize;
      }
   }
}

void GASNetAPI::put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, std::size_t size )
{
   if ( gasnet_mynode() != 0 ) fatal0("Error, cant use ::put from node != than 0"); 
   _pinnedAllocatorsLocks[ remoteNode ]->acquire();
   void *tmp = _pinnedAllocators[ remoteNode ]->allocate( size );
   _pinnedAllocatorsLocks[ remoteNode ]->release();
   _put( remoteNode, remoteAddr, localAddr, size, tmp );
}

//Lock getLock;
#ifndef GASNET_SEGMENT_EVERYTHING
Lock getLockGlobal;
#endif

void GASNetAPI::get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, std::size_t size )
{
   std::size_t sent = 0, thisReqSize;
   volatile int requestComplete = 0;
   //volatile int *reqPtr = &(requestComplete);

#ifndef GASNET_SEGMENT_EVERYTHING
   getLockGlobal.acquire();
   void *addr = _thisNodeSegment->allocate( size );
   getLockGlobal.release();
#else
   void *addr = localAddr;
#endif

   //fprintf(stderr, "get ( dest=%d, remote=%p, locla=%p, size=%ld, localtmp=%p, maxreq=%ld)\n", remoteNode, (void *) remoteAddr, localAddr, size, addr, gasnet_AMMaxLongRequest());
   //message("get ( dest=" << remoteNode <<", remote="<<(void *) remoteAddr<<", locla=" << localAddr << ", size="<< size <<", localtmp="<<addr<<", maxreq="<<gasnet_AMMaxLongRequest() << ", waitObj=" << reqPtr );
   //fprintf(stderr, "my req obj is %p\n", reqPtr);

   while ( sent < size )
   {
      thisReqSize = ( ( size - sent ) <= gasnet_AMMaxLongReply() ) ? size - sent : gasnet_AMMaxLongReply();

      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( ( ( sent + thisReqSize ) == size ) ? &requestComplete : NULL ) ; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent ( NANOS_XFER_GET, id, sizeKey, xferSize, remoteNode ); )

      //fprintf(stderr, "n:%d send get req to node %d(src=%p, dst=%p localtag=%p, size=%ld)\n", gasnet_mynode(), remoteNode, (void *) remoteAddr, (void *) ( ( uintptr_t ) ( ( uintptr_t ) addr ) + sent ), localAddr, thisReqSize  );
      if ( gasnet_AMRequestShort10( remoteNode, 211,
               ARG_LO( ( ( uintptr_t ) ( ( uintptr_t ) addr ) + sent )  ),
               ARG_HI( ( ( uintptr_t ) ( ( uintptr_t ) addr ) + sent )  ),
               ARG_LO( remoteAddr + sent ),
               ARG_HI( remoteAddr + sent ),
               ARG_LO( remoteAddr ),
               ARG_HI( remoteAddr ),
               ARG_LO( thisReqSize ),
               ARG_HI( thisReqSize ),
               ARG_LO( (uintptr_t) (( ( sent + thisReqSize ) == size ) ? &requestComplete : NULL )),
               ARG_HI( (uintptr_t) (( ( sent + thisReqSize ) == size ) ? &requestComplete : NULL )) ) != GASNET_OK)
      {
         fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
      }
      sent += thisReqSize;
   }
   //message("after gasnet req get ( dest=" << remoteNode <<", remote="<<(void *) remoteAddr<<", locla=" << localAddr << ", size="<< size <<", localtmp="<<addr<<", maxreq="<<gasnet_AMMaxLongRequest() << ", waitObj=" << reqPtr );

   //fprintf(stderr, "my req obj is %p\n", reqPtr);
   while ( requestComplete == 0 )
      poll();
   rxBytes += size;
   _totalBytes += size;
   //fprintf(stderr, "my req obj is %p\n", reqPtr);

//message("pre inval addr " << (void *) localAddr  );
   DirectoryEntry *ent = _masterDir->findEntry( (uint64_t) localAddr );
   if (ent != NULL) 
   {
      if (ent->getOwner() != NULL )
      {
         if ( !ent->isInvalidated() )
         {
            std::list<uint64_t> tagsToInvalidate;
            tagsToInvalidate.push_back( ( uint64_t ) localAddr );
            _masterDir->synchronizeHostSoft( tagsToInvalidate );
         }
      }
   }

#ifndef GASNET_SEGMENT_EVERYTHING
   // copy the data to the correct addr;
   ::memcpy( localAddr, addr, size );
   getLockGlobal.acquire();
   _thisNodeSegment->free( addr );
   getLockGlobal.release();
   //fprintf(stderr, "get data is %f\n", *((float *)localAddr) );
#endif
}

void GASNetAPI::malloc ( unsigned int remoteNode, std::size_t size, void * waitObjAddr )
{
   //message0("Requesting alloc of " << size << " bytes (" << (void *) size << ") to node " << remoteNode );
   if (gasnet_AMRequestShort4( remoteNode, 207,
            ARG_LO( size ), ARG_HI( size ),
            ARG_LO( waitObjAddr ), ARG_HI( waitObjAddr ) ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
   }
} 

void GASNetAPI::memRealloc ( unsigned int remoteNode, void *oldAddr, std::size_t oldSize, void *newAddr, std::size_t newSize )
{
   if (gasnet_AMRequestShort8( remoteNode, 217,
            ARG_LO( oldAddr ), ARG_HI( oldAddr ),
            ARG_LO( oldSize ), ARG_HI( oldSize ),
            ARG_LO( newAddr ), ARG_HI( newAddr ),
            ARG_LO( newSize ), ARG_HI( newSize ) ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
   }
}

void GASNetAPI::memFree ( unsigned int remoteNode, void *addr )
{
   if (gasnet_AMRequestShort2( remoteNode, 216,
            ARG_LO( addr ), ARG_HI( addr ) ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
   }
   //std::cerr << "FIXME: I should do something in GASNetAPI::memFree." << std::endl;
}

void GASNetAPI::nodeBarrier()
{
   gasnet_barrier_notify( 0, GASNET_BARRIERFLAG_ANONYMOUS );
   gasnet_barrier_wait( 0, GASNET_BARRIERFLAG_ANONYMOUS );
}

void GASNetAPI::sendMyHostName( unsigned int dest )
{
   const char *masterHostname = sys.getNetwork()->getMasterHostname();

   if ( masterHostname == NULL )
      fprintf(stderr, "Error, master hostname not set!\n" );

   if ( gasnet_AMRequestMedium0( dest, 209, ( void * ) masterHostname, strlen( masterHostname ) ) != GASNET_OK )
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest );
   }
}

void GASNetAPI::sendRequestPut( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, std::size_t len )
{
   _sentDataLock.acquire();
   //if ( dataDest == 3 )   fprintf(stderr, "added %p to sent data for node %d\n", (void *)dstAddr, dataDest);
   _sentData[ dataDest ]->insert( dstAddr );
   _sentDataLock.release();
   _totalBytes += len;
   sendWaitForRequestPut( dataDest, dstAddr );
   _pinnedAllocatorsLocks[ dataDest ]->acquire();
   void *tmpBuffer = _pinnedAllocators[ dataDest ]->allocate( len );
   _pinnedAllocatorsLocks[ dataDest ]->release();
   //if ( dataDest == 3 )fprintf(stderr, "master requesting to node %d to send addr %p to node %d addr %p\n", dest, (void *)origAddr, dataDest, (void *)dstAddr);
   if ( gasnet_AMRequestShort9( dest, 214,
            ARG_LO( dstAddr ), ARG_HI( dstAddr ),
            ARG_LO( ( ( uintptr_t ) origAddr ) ), ARG_HI( ( ( uintptr_t ) origAddr ) ),
            ARG_LO( ( ( uintptr_t ) tmpBuffer ) ), ARG_HI( ( ( uintptr_t ) tmpBuffer ) ),
	    ARG_LO( len ),
	    ARG_HI( len ),
            dataDest ) != GASNET_OK )

   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GASNetAPI::setMasterDirectory(Directory *dir)
{
   _masterDir = dir;
}

void GASNetAPI::sendWaitForRequestPut( unsigned int dest, uint64_t addr )
{
   if ( gasnet_AMRequestShort2( dest, 218,
            ARG_LO( addr ), ARG_HI( addr ) ) != GASNET_OK )
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
	//message("sending sendWaitForRequestPut( to node " << dest << ", addr "<< (void *) addr <<").");
}

void GASNetAPI::sendFreeTmpBuffer( void *addr )
{
   if ( gasnet_AMRequestShort2( 0, 219,
            ARG_LO( addr ), ARG_HI( addr ) ) != GASNET_OK )
   {
      fprintf(stderr, "gasnet: Error sending a message to node 0.\n");
   }
}

std::size_t GASNetAPI::getRxBytes()
{
   return rxBytes;
}

std::size_t GASNetAPI::getTxBytes()
{
   return txBytes;
}

std::size_t GASNetAPI::getTotalBytes()
{
   return _totalBytes;
}
