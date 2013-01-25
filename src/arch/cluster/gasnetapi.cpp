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
#include "clusterdevice_decl.hpp"
#include "instrumentation.hpp"
#include "atomic_decl.hpp"
#include "regiondirectory.hpp"
#include "osallocator_decl.hpp"
#include "requestqueue.hpp"
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

#define VERBOSE_AM( x )

#ifdef GPU_DEV
//FIXME: GPU Support
void * local_nanos_gpu_factory( void *args );
void * local_nanos_gpu_factory( void *args )
{
   nanos_smp_args_t *smp = ( nanos_smp_args_t * ) args;
   return ( void * )new ext::GPUDD( smp->outline );
   //if ( prealloc != NULL )
   //{
   //   return ( void * )new (prealloc) ext::GPUDD( smp->outline );
   //}
   //else
   //{
   //   return ( void * ) new ext::GPUDD( smp->outline );
   //}
}
#endif
void * local_nanos_smp_factory( void *args );
void * local_nanos_smp_factory( void *args )
{
   nanos_smp_args_t *smp = ( nanos_smp_args_t * ) args;
   return ( void * )new ext::SMPDD( smp->outline );
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
NewRegionDirectory * GASNetAPI::_newMasterDir;
#ifndef GASNET_SEGMENT_EVERYTHING
SimpleAllocator * GASNetAPI::_thisNodeSegment;
#endif
SimpleAllocator * GASNetAPI::_packSegment;

//Lock GASNetAPI::_sentDataLock;
//std::vector<std::set<uint64_t> *> GASNetAPI::_sentData;

RequestQueue< struct GASNetAPI::putReqDesc > GASNetAPI::_putReqs;
std::list<struct GASNetAPI::putReqDesc * > GASNetAPI::_delayedPutReqs; //this should be a map
Lock GASNetAPI::_delayedPutReqsLock;
// NOT YET IMPLEMENTED std::list<struct GASNetAPI::delayedGetDesc * > GASNetAPI::_delayedGets; //Not actually used? implementation not finished
// NOT YET IMPLEMENTED Lock GASNetAPI::_delayedGetsLock;

std::set<void *> GASNetAPI::_receivedUnmatchedPutRequests;
std::set<void *> GASNetAPI::_waitingPutRequests;
Lock GASNetAPI::_waitingPutRequestsLock;
std::size_t GASNetAPI::rxBytes = 0;
std::size_t GASNetAPI::txBytes = 0;
std::size_t GASNetAPI::_totalBytes = 0;
std::vector< SimpleAllocator * > GASNetAPI::_pinnedAllocators;
std::vector< Lock * > GASNetAPI::_pinnedAllocatorsLocks;
std::list< std::pair< void *, WD const * > > GASNetAPI::_freeBufferReqs;
Lock GASNetAPI::_freeBufferReqsLock;

Lock GASNetAPI::_workDoneReqsLock;
std::list< std::pair<void *, unsigned int> > GASNetAPI::_workDoneReqs;
Atomic<unsigned int> *GASNetAPI::_seqN;
#if 0
Atomic<unsigned int> GASNetAPI::_recvSeqN (0);
Lock GASNetAPI::_deferredWorkReqsLock;
std::list< std::pair< unsigned int, std::pair<WD *, std::size_t > > > GASNetAPI::_deferredWorkReqs;
GASNetAPI::ReceivedWDData GASNetAPI::_recvWdData;
GASNetAPI::SentWDData GASNetAPI::_sentWdData;
#endif

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

//void GASNetAPI::processWork2(std::size_t expectedData, WD *delayedWD, unsigned int delayedSeq) {
//   if ( _recvSeqN.value() == delayedSeq )
//   {
//      _recvWdData.addWD( delayedWD->getId(), delayedWD, expectedData );
//      checkDeferredWorkReqs();
//   } else { //not expected seq number, enqueue
//      _deferredWorkReqsLock.acquire();
//      _deferredWorkReqs.push_back( std::pair< unsigned int, std::pair < WD*, std::size_t > >( delayedSeq, std::pair< WD*, std::size_t > ( delayedWD, expectedData ) ) );
//      _deferredWorkReqsLock.release();
//   }
//}

void GASNetAPI::print_copies( WD const *wd, int deps )
{
#if 1
   unsigned int i;
   fprintf(stderr, "node %d submit slave %s wd %d with %d deps, copies are: ", gasnet_mynode(), /*(((WG*) wd)->getParent() == (WG*) GASNetAPI::_rwgGPU ? "GPU" : "SMP")*/"n/a", wd->getId(), deps );
   for ( i = 0; i < wd->getNumCopies(); i++)
      fprintf(stderr, "%s%s:%p ", ( wd->getCopies()[i].isInput() ? "r" : "-" ), ( wd->getCopies()[i].isOutput() ? "w" : "-" ), (void *) wd->getCopies()[i].getAddress() );
   fprintf(stderr, "\n");
#endif
}

// NOT YET IMPLEMENTED void GASNetAPI::delayAmGet( unsigned int dest, void *origTag, void *origAddr, void *destAddr, std::size_t len, std::size_t count, std::size_t ld, void *waitObj ) {
// NOT YET IMPLEMENTED    struct delayedGetDesc *dgd = NEW struct delayedGetDesc();
// NOT YET IMPLEMENTED    dgd->dest = dest;
// NOT YET IMPLEMENTED    dgd->origTag = origTag;
// NOT YET IMPLEMENTED    dgd->origAddr = origAddr;
// NOT YET IMPLEMENTED    dgd->destAddr = destAddr;
// NOT YET IMPLEMENTED    dgd->len = len;
// NOT YET IMPLEMENTED    dgd->count = count;
// NOT YET IMPLEMENTED    dgd->ld = ld;
// NOT YET IMPLEMENTED    dgd->waitObj = waitObj;
// NOT YET IMPLEMENTED 
// NOT YET IMPLEMENTED    _delayedGetsLock.acquire();
// NOT YET IMPLEMENTED    _delayedGets.push_back( dgd );
// NOT YET IMPLEMENTED    _delayedGetsLock.release();
// NOT YET IMPLEMENTED }

void GASNetAPI::enqueuePutReq( unsigned int dest, void *origAddr, void *destAddr, std::size_t len, std::size_t count, std::size_t ld, void *tmpBuffer, unsigned int wdId, WD const *wd )
{
   struct putReqDesc *prd = NEW struct putReqDesc();
   prd->dest = dest;
   prd->origAddr = origAddr;
   prd->destAddr = destAddr;
   prd->len = len;
   prd->count = count;
   prd->ld = ld;
   prd->tmpBuffer = tmpBuffer;
   prd->wdId = wdId;
   prd->wd = wd;

   _putReqs.add( prd );
}

void GASNetAPI::enqueueDelayedPutReq( unsigned int dest, void *origAddr, void *destAddr, std::size_t len, std::size_t count, std::size_t ld, void *tmpBuffer, unsigned int wdId, WD const *wd )
{
   struct putReqDesc *prd = NEW struct putReqDesc();
   prd->dest = dest;
   prd->origAddr = origAddr;
   prd->destAddr = destAddr;
   prd->len = len;
   prd->count = count;
   prd->ld = ld;
   prd->tmpBuffer = tmpBuffer;
   prd->wdId = wdId;
   prd->wd = wd;

   _delayedPutReqsLock.acquire();
   _delayedPutReqs.push_back( prd );
   _delayedPutReqsLock.release();
}

void GASNetAPI::checkForPutReqs()
{
   struct putReqDesc *prd = _putReqs.tryFetch();
   if ( prd != NULL ) {
      if ( prd->ld == 0 ) {
         NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-in") );
         NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER_IN, key, (nanos_event_value_t) prd->len) );
         _put( prd->dest, (uint64_t) prd->destAddr, prd->origAddr, prd->len, prd->tmpBuffer, prd->wdId, *prd->wd );
         NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
      } else {
         char *localPack, *origAddrPtr = (char *) prd->origAddr;

         NANOS_INSTRUMENT( InstrumentState inst2(NANOS_STRIDED_COPY_PACK); );
         _packSegment->lock();
         localPack = ( char * ) _packSegment->allocate( prd->len * prd->count );
         if ( localPack == NULL ) { fprintf(stderr, "ERROR!!! could not get an addr to pack strided data\n" ); }
         _packSegment->unlock();

         for ( unsigned int i = 0; i < prd->count; i += 1 ) {
            memcpy( &localPack[ i * prd->len ], &origAddrPtr[ i * prd->ld ], prd->len );
         }
         NANOS_INSTRUMENT( inst2.close(); );

         if ( prd->tmpBuffer == NULL ) std::cerr <<" smt goes wronf!"<<std::endl;
         _putStrided1D( prd->dest, (uint64_t) prd->destAddr, prd->origAddr, localPack, prd->len, prd->count, prd->ld, prd->tmpBuffer, prd->wdId, *prd->wd );

         _packSegment->lock();
         _packSegment->free( localPack );
         _packSegment->unlock();
      }
      delete prd;
   }
}

void GASNetAPI::enqueueFreeBufferNotify( void *tmpBuffer, WD const* wd )
{
   _freeBufferReqsLock.acquire();
   _freeBufferReqs.push_back( std::make_pair( tmpBuffer, wd ) );
   _freeBufferReqsLock.release();
}

void GASNetAPI::checkForFreeBufferReqs()
{
   std::pair< void *, WD const * > addrWd;
   if ( _freeBufferReqsLock.tryAcquire() )
   {
      if ( !_freeBufferReqs.empty() )
      {
         addrWd = _freeBufferReqs.front();
         _freeBufferReqs.pop_front();
      }
      _freeBufferReqsLock.release();
   }
   if ( addrWd.first != NULL )
      sendFreeTmpBuffer( addrWd.first, addrWd.second );
}

void GASNetAPI::checkWorkDoneReqs()
{
   std::pair<void *, unsigned int> rwd(NULL, 0);
   if ( _workDoneReqsLock.tryAcquire() )
   {
      if ( !_workDoneReqs.empty() )
      {
         rwd = _workDoneReqs.front();
         _workDoneReqs.pop_front();
      }
      _workDoneReqsLock.release();
   }
   if ( rwd.first != NULL )
      _sendWorkDoneMsg( 0, rwd.first, rwd.second );
}
#if 0
void GASNetAPI::checkDeferredWorkReqs()
{
   std::pair<unsigned int, std::pair< WD *, std::size_t > > dwd ( 0, std::pair< WD *, std::size_t > ( NULL, 0 )  );
   if ( _deferredWorkReqsLock.tryAcquire() )
   {
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
            _recvWdData.addWD( dwd.second.first->getId(), dwd.second.first, dwd.second.second );
            checkDeferredWorkReqs();
         } else {
            _deferredWorkReqs.push_back( dwd );
            _deferredWorkReqsLock.release();
         }
      } else {
         _deferredWorkReqsLock.release();
      }
   }
}
#endif

void GASNetAPI::amFinalize(gasnet_token_t token)
{
   gasnet_node_t src_node;
   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error obtaining node information.\n");
   }
   //gasnet_AMReplyShort0(token, 204);
   sys.stopFirstThread();
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amFinalizeReply(gasnet_token_t token)
{
   gasnet_node_t src_node;
   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error obtaining node information.\n");
   }
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amWork(gasnet_token_t token, void *arg, std::size_t argSize,
      gasnet_handlerarg_t workLo,
      gasnet_handlerarg_t workHi,
      gasnet_handlerarg_t xlateLo,
      gasnet_handlerarg_t xlateHi,
      gasnet_handlerarg_t rmwdLo,
      gasnet_handlerarg_t rmwdHi,
      gasnet_handlerarg_t expectedDataLo,
      gasnet_handlerarg_t expectedDataHi,
      unsigned int dataSize, unsigned int wdId, unsigned int numPe, int arch, unsigned int seq )
{
   void (*work)( void *) = (void (*)(void *)) MERGE_ARG( workHi, workLo );
   void (*xlate)( void *, void *) = (void (*)(void *, void *)) MERGE_ARG( xlateHi, xlateLo );
   void *rmwd = (void *) MERGE_ARG( rmwdHi, rmwdLo );
   std::size_t expectedData = (std::size_t) MERGE_ARG( expectedDataHi, expectedDataLo );
   gasnet_node_t src_node;
   unsigned int i;
   WorkGroup *rwg;

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
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

   int num_dimensions = *((int *) &work_data[ dataSize + sizeof( int ) + numCopies * sizeof( CopyData ) ]);
   nanos_region_dimension_internal_t *dimensions = NULL;
   nanos_region_dimension_internal_t **dimensions_ptr = ( num_dimensions > 0 ) ? &dimensions : NULL ;

   nanos_device_t newDeviceSMP = { local_nanos_smp_factory, (void *) &smp_args } ;
#ifdef GPU_DEV
   nanos_device_t newDeviceGPU = { local_nanos_gpu_factory, (void *) &smp_args } ;
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
   sys.createWD( &localWD, (std::size_t) 1, devPtr, (std::size_t) dataSize, (int) ( sizeof(void *) ), (void **) &data, (WG *)rwg, (nanos_wd_props_t *) NULL, (nanos_wd_dyn_props_t *) NULL, (std::size_t) numCopies, newCopiesPtr, num_dimensions, dimensions_ptr, xlate );

   std::memcpy(data, work_data, dataSize);

   //unsigned int numDeps = *( ( int * ) &work_data[ dataSize + sizeof( int ) + numCopies * sizeof( CopyData ) + sizeof( int ) + num_dimensions * sizeof( nanos_region_dimension_t ) ] );
   //uint64_t *depTags = ( ( uint64_t * ) &work_data[ dataSize + sizeof( int ) + numCopies * sizeof( CopyData ) + sizeof( int ) + num_dimensions * sizeof( nanos_region_dimension_t ) + sizeof( int ) ] );

   // Set copies and dimensions, getDimensions() returns an index here, instead of a pointer,
   // the index is the position inside the dimension array that must be set as the base address for the dimensions
   CopyData *recvCopies = ( ( CopyData *) &work_data[ dataSize + sizeof( int ) ] );
   nanos_region_dimension_t *recvDimensions = ( ( nanos_region_dimension_t * ) &work_data[ dataSize + sizeof( int ) + numCopies * sizeof( CopyData ) + sizeof(int) ] );
   memcpy( *dimensions_ptr, recvDimensions, num_dimensions * sizeof(nanos_region_dimension_t) );
   for (i = 0; i < numCopies; i += 1)
   {
      new ( &newCopies[i] ) CopyData( recvCopies[i] );
      newCopies[i].setDimensions( (*dimensions_ptr) + ( ( uintptr_t ) recvCopies[i].getDimensions() ) );
   }

   localWD->setId( wdId );
   localWD->setRemoteAddr( rmwd );
   {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) wdId)  )  ; )
      NANOS_INSTRUMENT ( instr->createDeferredPtPEnd ( *localWD, NANOS_WD_REMOTE, id, 0, NULL, NULL, 0 ); )
   }

   _net->notifyWorkArrival(expectedData, localWD, seq);

   delete[] work_data;
   work_data = NULL;
   work_data_len = 0;
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amWorkData(gasnet_token_t token, void *buff, std::size_t len,
      gasnet_handlerarg_t msgNum,
      gasnet_handlerarg_t totalLenLo,
      gasnet_handlerarg_t totalLenHi)
{
   gasnet_node_t src_node;
   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error obtaining node information.\n");
   }

   std::cerr<<"UNSUPPORTED FOR NOW"<<std::endl;
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amWorkDone( gasnet_token_t token, gasnet_handlerarg_t addrLo, gasnet_handlerarg_t addrHi, gasnet_handlerarg_t peId )
{
   gasnet_node_t src_node;
   void * addr = (void *) MERGE_ARG( addrHi, addrLo );
   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( addr ) ; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_AM_WORK_DONE, id, 0, 0, src_node ); )

   sys.getNetwork()->notifyWorkDone( src_node, addr, peId );
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amMalloc( gasnet_token_t token, gasnet_handlerarg_t sizeLo, gasnet_handlerarg_t sizeHi,
      gasnet_handlerarg_t waitObjAddrLo, gasnet_handlerarg_t waitObjAddrHi )
{
   gasnet_node_t src_node;
   void *addr = NULL; //volatile int *ptr;
   std::size_t size = ( std::size_t ) MERGE_ARG( sizeHi, sizeLo );
   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   OSAllocator a;
   addr = a.allocate( size );
   if ( addr == NULL )  {
      std::cerr << "ERROR at amMalloc" << std::endl;
   }
   
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
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amMallocReply( gasnet_token_t token, gasnet_handlerarg_t addrLo, gasnet_handlerarg_t addrHi,
      gasnet_handlerarg_t waitObjAddrLo, gasnet_handlerarg_t waitObjAddrHi )
{
   void * addr = ( void * ) MERGE_ARG( addrHi, addrLo );
   Network::mallocWaitObj *request = ( Network::mallocWaitObj * ) MERGE_ARG( waitObjAddrHi, waitObjAddrLo );
   gasnet_node_t src_node;
   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   sys.getNetwork()->notifyMalloc( src_node, addr, request );
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amFree( gasnet_token_t token, gasnet_handlerarg_t addrLo, gasnet_handlerarg_t addrHi )
{
   void * addr = (void *) MERGE_ARG( addrHi, addrLo );
   gasnet_node_t src_node;

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   free( addr );
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
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

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   std::memcpy( newAddr, oldAddr, oldSize );
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amMasterHostname( gasnet_token_t token, void *buff, std::size_t nbytes )
{
   gasnet_node_t src_node;
   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   /* for now we only allow this at node 0 */
   if ( src_node == 0 )
   {
      sys.getNetwork()->setMasterHostname( ( char  *) buff );
   }
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amPut( gasnet_token_t token,
      void *buf,
      std::size_t len,
      gasnet_handlerarg_t realAddrLo,
      gasnet_handlerarg_t realAddrHi,
      gasnet_handlerarg_t realTagLo,
      gasnet_handlerarg_t realTagHi,
      gasnet_handlerarg_t totalLenLo,
      gasnet_handlerarg_t totalLenHi,
      gasnet_handlerarg_t wdId,
      gasnet_handlerarg_t wdLo,
      gasnet_handlerarg_t wdHi,
      gasnet_handlerarg_t firstMsg,
      gasnet_handlerarg_t lastMsg ) 
{
   gasnet_node_t src_node;
   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   void *realAddr = ( int * ) MERGE_ARG( realAddrHi, realAddrLo );
   void *realTag = ( int * ) MERGE_ARG( realTagHi, realTagLo );
   WD *wd = ( WD * ) MERGE_ARG( wdHi, wdLo );
   std::size_t totalLen = ( std::size_t ) MERGE_ARG( totalLenHi, totalLenLo );

   rxBytes += len;

   
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = len; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( buf ) ; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_PUT, id, sizeKey, xferSize, src_node ); )

   if ( firstMsg )
   {
      //fprintf(stderr, "[%d] BEGIN amPut to node %d, from %d, local (maybe tmp) addr %p, realTag %p, realAddr (local) %p  data is %f\n", myThread->getId(),  gasnet_mynode(), src_node, buf, realTag, realAddr, *((float *)buf));
      if ( sys.getNetwork()->doIHaveToCheckForDataInOtherAddressSpaces() ) {
         invalidateDataFromDevice( (uint64_t) realTag, totalLen );
         //fprintf(stderr, "im node %d and im invalidateDataFromDevice @ %s, addr %p, %f\n", gasnet_mynode(), __FUNCTION__, realTag, *((double *)buf));
      }
   }
   if ( realAddr != NULL )
   {
      ::memcpy( realAddr, buf, len );
   }
   if ( lastMsg )
   {
      uintptr_t localAddr =  ( ( uintptr_t ) buf ) - ( ( uintptr_t ) realAddr - ( uintptr_t ) realTag );
      enqueueFreeBufferNotify( ( void * ) localAddr, wd );
      _net->notifyPutArrival( wdId, totalLen );
      if ( src_node != 0 ) { /* check for delayed putReqs or gets */
         _waitingPutRequestsLock.acquire();
         std::set<void *>::iterator it;
         if ( ( it = _waitingPutRequests.find( realTag ) ) != _waitingPutRequests.end() )
         {
            void *destAddr = *it;
            _waitingPutRequests.erase( it );
            _delayedPutReqsLock.acquire();
            if ( !_delayedPutReqs.empty() ) {
               for ( std::list<struct GASNetAPI::putReqDesc *>::iterator putReqsIt = _delayedPutReqs.begin(); putReqsIt != _delayedPutReqs.end(); putReqsIt++ ) {
                  if ( (*putReqsIt)->origAddr == destAddr ) {
                     _putReqs.add( *putReqsIt );
                  }
               }
            }
            _delayedPutReqsLock.release();
         }
         else
         {
            _receivedUnmatchedPutRequests.insert( realTag );
         }
         _waitingPutRequestsLock.release();
      }
   }
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amPutStrided1D( gasnet_token_t token,
      void *buf,
      std::size_t len,
      gasnet_handlerarg_t realTagLo,
      gasnet_handlerarg_t realTagHi,
      gasnet_handlerarg_t sizeLo,
      gasnet_handlerarg_t sizeHi,
      gasnet_handlerarg_t count,
      gasnet_handlerarg_t ld,
      gasnet_handlerarg_t wdId,
      gasnet_handlerarg_t wdLo,
      gasnet_handlerarg_t wdHi,
      gasnet_handlerarg_t firstMsg,
      gasnet_handlerarg_t lastMsg ) 
{
   gasnet_node_t src_node;
   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   void *realTag = ( int * ) MERGE_ARG( realTagHi, realTagLo );
   std::size_t size = ( std::size_t ) MERGE_ARG( sizeHi, sizeLo );
   std::size_t totalLen = size * count;
   WD *wd = ( WD * ) MERGE_ARG( wdHi, wdLo );

   rxBytes += len;

   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = len; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( buf ) ; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_PUT, id, sizeKey, xferSize, src_node ); )

   if ( firstMsg )
   {
      if ( sys.getNetwork()->doIHaveToCheckForDataInOtherAddressSpaces() ) {
         std::cerr <<"Unsupported for now"<<std::endl;
         //invalidateDataFromDevice( (uint64_t) realTag, totalLen );
      }
   }
   if ( lastMsg )
   {
      char* realAddrPtr = (char *) realTag;
      char* localAddrPtr = ( (char *) ( ( ( uintptr_t ) buf ) + ( ( uintptr_t ) len ) - ( uintptr_t ) totalLen ) );
      NANOS_INSTRUMENT( InstrumentState inst2(NANOS_STRIDED_COPY_UNPACK); );
      for ( int i = 0; i < count; i += 1 ) {
         ::memcpy( &realAddrPtr[ i * ld ], &localAddrPtr[ i * size ], size );
      }
      NANOS_INSTRUMENT( inst2.close(); );
      //send notify to node 
      // compute the original local address
      uintptr_t localAddr = ( ( uintptr_t ) buf ) + ( ( uintptr_t ) len ) - ( uintptr_t ) totalLen;
      enqueueFreeBufferNotify( ( void * ) localAddr, wd );
      _net->notifyPutArrival( wdId, totalLen );

      if ( src_node != 0 ) { //only check if it is a put from a request put
         _waitingPutRequestsLock.acquire();
         std::set<void *>::iterator it;
         if ( ( it = _waitingPutRequests.find( realTag ) ) != _waitingPutRequests.end() ) {
            void *destAddr = *it;
            _waitingPutRequests.erase( it );
            _delayedPutReqsLock.acquire();
            if ( !_delayedPutReqs.empty() ) {
               for ( std::list<struct GASNetAPI::putReqDesc *>::iterator putReqsIt = _delayedPutReqs.begin(); putReqsIt != _delayedPutReqs.end(); putReqsIt++ ) {
                  if ( (*putReqsIt)->origAddr == destAddr ) {
                     fprintf(stderr, "not empty!... %p and found!\n", destAddr);
                     _putReqs.add( *putReqsIt );
                  }
               }
            }
            _delayedPutReqsLock.release();
         } else {
            //message("addr " << (void *) realTag << " NOT found! unmatched requests.. im node " << gasnet_mynode() << " for wd " << wdId);
            _receivedUnmatchedPutRequests.insert( realTag );
         }
         _waitingPutRequestsLock.release();
      }

   }
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amGetReply( gasnet_token_t token,
      void *buf,
      std::size_t len,
      gasnet_handlerarg_t waitObjLo,
      gasnet_handlerarg_t waitObjHi)
{
   gasnet_node_t src_node;
   ClusterDevice::GetRequest *waitObj = ( ClusterDevice::GetRequest * ) MERGE_ARG( waitObjHi, waitObjLo );

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   //fprintf(stderr, "get reply from %d: data is %d waitObj %p\n", src_node , *((int *)buf), waitObj);
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = len; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) waitObj; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent ( NANOS_XFER_GET, id, sizeKey, xferSize, src_node ); )

   if ( waitObj != NULL )
   {
      //waitObj->_complete = 1;
      waitObj->complete();
   }
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amGetReplyStrided1D( gasnet_token_t token,
      void *buf,
      std::size_t len,
      gasnet_handlerarg_t waitObjLo,
      gasnet_handlerarg_t waitObjHi)
{
   gasnet_node_t src_node;
   ClusterDevice::GetRequest *waitObj = ( ClusterDevice::GetRequest * ) MERGE_ARG( waitObjHi, waitObjLo );

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = len; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) waitObj; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent ( NANOS_XFER_GET, id, sizeKey, xferSize, src_node ); )

   if ( waitObj != NULL )
   {
      //   waitObj->clear();
      waitObj->complete();
   }
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
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
      gasnet_handlerarg_t totalLenLo,
      gasnet_handlerarg_t totalLenHi,
      gasnet_handlerarg_t waitObjLo,
      gasnet_handlerarg_t waitObjHi )
{
   gasnet_node_t src_node;
   void *origAddr = ( void * ) MERGE_ARG( origAddrHi, origAddrLo );
   void *destAddr = ( void * ) MERGE_ARG( destAddrHi, destAddrLo );
   void *tagAddr = ( void * ) MERGE_ARG( tagAddrHi, tagAddrLo );
   NANOS_INSTRUMENT ( int *waitObj = ( int * ) MERGE_ARG( waitObjHi, waitObjLo ); )
   std::size_t realLen = ( std::size_t ) MERGE_ARG( lenHi, lenLo );
   std::size_t totalLen = ( std::size_t ) MERGE_ARG( totalLenHi, totalLenLo );

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
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

      if ( sys.getNetwork()->doIHaveToCheckForDataInOtherAddressSpaces() ) {
         getDataFromDevice( (uint64_t) tagAddr, totalLen );
         //fprintf(stderr, "im node %d and im getDataFromDevice @ %s, sent data is %f, addr %p\n", gasnet_mynode(), __FUNCTION__, *data, tagAddr);
      }
   }
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
   }
   else
   {
      fprintf( stderr, "gasnet: Error, requested a GET of size > gasnet_AMMaxLongReply() bytes.\n" );
   }
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
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

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
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

   //NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   //NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   //NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   //NANOS_INSTRUMENT ( nanos_event_value_t xferSize = len; )
   //NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( destAddr ) ; )
   //NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_PUT, id, sizeKey, xferSize, src_node ); )
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
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
      gasnet_handlerarg_t wdId,
      gasnet_handlerarg_t wdLo,
      gasnet_handlerarg_t wdHi,
      gasnet_handlerarg_t dst )
{
   gasnet_node_t src_node;
   void *origAddr = ( void * ) MERGE_ARG( origAddrHi, origAddrLo );
   void *destAddr = ( void * ) MERGE_ARG( destAddrHi, destAddrLo );
   void *tmpBuffer = ( void * ) MERGE_ARG( tmpBufferHi, tmpBufferLo );
   std::size_t len = ( std::size_t ) MERGE_ARG( lenHi, lenLo );
   WD *wd = ( WD * ) MERGE_ARG( wdHi, wdLo );

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   _waitingPutRequestsLock.acquire();
   if ( _waitingPutRequests.find( origAddr ) != _waitingPutRequests.end() ) //we have to wait 
   {
      _waitingPutRequestsLock.release();
      enqueueDelayedPutReq( dst, origAddr, destAddr, len, 1, 0, tmpBuffer, wdId, wd );
   } else {
      _waitingPutRequestsLock.release();

      enqueuePutReq( dst, origAddr, destAddr, len, 1, 0, tmpBuffer, wdId, wd );
   }
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amRequestPutStrided1D( gasnet_token_t token,
      gasnet_handlerarg_t destAddrLo,
      gasnet_handlerarg_t destAddrHi,
      gasnet_handlerarg_t origAddrLo,
      gasnet_handlerarg_t origAddrHi,
      gasnet_handlerarg_t tmpBufferLo,
      gasnet_handlerarg_t tmpBufferHi,
      gasnet_handlerarg_t lenLo,
      gasnet_handlerarg_t lenHi,
      gasnet_handlerarg_t count,
      gasnet_handlerarg_t ld,
      gasnet_handlerarg_t wdId,
      gasnet_handlerarg_t wdLo,
      gasnet_handlerarg_t wdHi,
      gasnet_handlerarg_t dst )
{
   gasnet_node_t src_node;
   void *origAddr = ( void * ) MERGE_ARG( origAddrHi, origAddrLo );
   void *destAddr = ( void * ) MERGE_ARG( destAddrHi, destAddrLo );
   void *tmpBuffer = ( void * ) MERGE_ARG( tmpBufferHi, tmpBufferLo );
   std::size_t len = ( std::size_t ) MERGE_ARG( lenHi, lenLo );
   WD *wd = ( WD * ) MERGE_ARG( wdHi, wdLo );

   NANOS_INSTRUMENT( InstrumentState inst(NANOS_amRequestPutStrided1D); );

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( tmpBuffer ) ; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_PUT, id, sizeKey, xferSize, src_node ); )

   _waitingPutRequestsLock.acquire();
   if ( _waitingPutRequests.find( origAddr ) != _waitingPutRequests.end() ) //we have to wait 
   {
      _waitingPutRequestsLock.release();
      enqueueDelayedPutReq( dst, origAddr, destAddr, len, count, ld, tmpBuffer, wdId, wd );
   } else {
      _waitingPutRequestsLock.release();
      if ( tmpBuffer == NULL ) std::cerr << " mmm oops "<<std::endl;
      enqueuePutReq( dst, origAddr, destAddr, len, count, ld, tmpBuffer, wdId, wd );
   }
   NANOS_INSTRUMENT( inst.close(); );
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amWaitRequestPut( gasnet_token_t token, 
      gasnet_handlerarg_t addrLo,
      gasnet_handlerarg_t addrHi )
{
   void *addr = ( void * ) MERGE_ARG( addrHi, addrLo );
   std::set< void * >::iterator it;
   NANOS_INSTRUMENT( InstrumentState inst(NANOS_amWaitRequestPut); );

   gasnet_node_t src_node;
   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( addr ) ; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_WAIT_REQ_PUT, id, sizeKey, xferSize, src_node ); )

   _waitingPutRequestsLock.acquire();
   if ( _receivedUnmatchedPutRequests.empty() || ( it = _receivedUnmatchedPutRequests.find( addr ) ) == _receivedUnmatchedPutRequests.end() )
   {
      _waitingPutRequests.insert( addr );
   } 
   else
   {
      _receivedUnmatchedPutRequests.erase( it );
   }
   _waitingPutRequestsLock.release();
   NANOS_INSTRUMENT( inst.close(); );
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amFreeTmpBuffer( gasnet_token_t token,
      gasnet_handlerarg_t addrLo,
      gasnet_handlerarg_t addrHi,
      gasnet_handlerarg_t wdLo,
      gasnet_handlerarg_t wdHi ) 
{
   void *addr = ( void * ) MERGE_ARG( addrHi, addrLo );
   WD *wd = ( WD * ) MERGE_ARG( wdHi, wdLo );
   (void) wd;
   gasnet_node_t src_node;
   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   //std::cerr << __FUNCTION__ << " addr " << addr << " from "<< src_node << " allocator "<< (void *) _pinnedAllocators[ src_node ] << std::endl;
         NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
         NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
         NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
         NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
         NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( addr ) ; )
         NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_FREE_TMP_BUFF, id, sizeKey, xferSize, src_node ); )
   _pinnedAllocatorsLocks[ src_node ]->acquire();
   _pinnedAllocators[ src_node ]->free( addr );
   _pinnedAllocatorsLocks[ src_node ]->release();
   //fprintf(stderr, "I CALL NOTIFY SCHED for wd %d\n", wd->getId() );
   // TODO i think this segfaults even when notify func has not been set
   // wd->notifyCopy();
   
   // XXX call notify copy wd->
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amGetStrided1D( gasnet_token_t token,
      gasnet_handlerarg_t destAddrLo,
      gasnet_handlerarg_t destAddrHi,
      gasnet_handlerarg_t origAddrLo,
      gasnet_handlerarg_t origAddrHi,
      gasnet_handlerarg_t origTagLo,
      gasnet_handlerarg_t origTagHi,
      gasnet_handlerarg_t lenLo,
      gasnet_handlerarg_t lenHi,
      gasnet_handlerarg_t count,
      gasnet_handlerarg_t ld,
      gasnet_handlerarg_t waitObjLo,
      gasnet_handlerarg_t waitObjHi )
{
   gasnet_node_t src_node;
   void *origAddr = ( void * ) MERGE_ARG( origAddrHi, origAddrLo );
   void *origTag = ( void * ) MERGE_ARG( origTagHi, origTagLo );
   void *destAddr = ( void * ) MERGE_ARG( destAddrHi, destAddrLo );
   NANOS_INSTRUMENT ( int *waitObj = ( int * ) MERGE_ARG( waitObjHi, waitObjLo ); )
   //void *waitObj = ( void * ) MERGE_ARG( waitObjHi, waitObjLo );
   std::size_t len = ( std::size_t ) MERGE_ARG( lenHi, lenLo );
   std::size_t totalLen = ( std::size_t ) len * count; 
   char *packAddr;

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   txBytes += totalLen;

   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) waitObj; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent ( NANOS_XFER_GET, id, sizeKey, xferSize, src_node ); )


   //fprintf(stderr, "n:%d thd %d am_xfer_get: srcAddr=%p, srcHi=%p, srcLo=%p, dstAddr=%p, dstHi=%p, dstLo=%p res=%f, waitObj=%p\n", gasnet_mynode(), myThread->getId(), origAddr, (void *)origAddrHi, (void *)origAddrLo, destAddr, (void*)destAddrHi, (void*)destAddrLo, *((float*)origAddr), /*waitObj*/(void *)NULL );
   _waitingPutRequestsLock.acquire();
   if ( _waitingPutRequests.find( origTag ) != _waitingPutRequests.end() ) //we have to wait 
   {
      _waitingPutRequestsLock.release();
      message("WARNING: origTag "<< origTag << " this amGetStrided SHOULD BE DELAYED!!! FIXME");
      //delayAmGet( src_node, origTag, origAddr, destAddr, len, count, ld, waitObj ); //NOT YET IMPLEMENTED
   } else {
      _waitingPutRequestsLock.release();

      //if ( sys.getNetwork()->doIHaveToCheckForDataInOtherAddressSpaces() ) {
      //   getDataFromDevice( (uint64_t) tagAddr, totalLen );
      //}
      NANOS_INSTRUMENT( InstrumentState inst2(NANOS_STRIDED_COPY_PACK); );
      char *origAddrPtr = ( char * ) origAddr;
      //getpack addr 
      _packSegment->lock();
      packAddr = ( char * ) _packSegment->allocate( totalLen );
      _packSegment->unlock();
      //pack it
      //fprintf(stderr, "packing addr %p to buff %p, len %ld\n", origAddrPtr, packAddr, totalLen);
      for ( unsigned int i = 0; i < ( unsigned int ) count; i += 1 ) {
         memcpy( &packAddr[ i * len ], &origAddrPtr[ i * ld ], len );
      }
      NANOS_INSTRUMENT( inst2.close(); );

      NANOS_INSTRUMENT ( xferSize = totalLen; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_GET, id, sizeKey, xferSize, src_node ); )

      if ( gasnet_AMReplyLong2( token, 212, /* TODO this is pack addr */ packAddr, totalLen, destAddr, waitObjLo, waitObjHi ) != GASNET_OK )
      {
         fprintf( stderr, "gasnet: Error sending reply msg.\n" );
      }
      _packSegment->lock();
      _packSegment->free( packAddr );
      _packSegment->unlock();
   }
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
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
      { 219, (void (*)()) amFreeTmpBuffer },
      { 220, (void (*)()) amPutStrided1D },
      { 221, (void (*)()) amGetStrided1D },
      { 222, (void (*)()) amRequestPutStrided1D },
      { 223, (void (*)()) amGetReplyStrided1D }
   };

   gasnet_init( &my_argc, &my_argv );

   segSize = gasnet_getMaxLocalSegmentSize();

   gasnet_attach( htable, sizeof( htable ) / sizeof( gasnet_handlerentry_t ), segSize, 0);

   _net->setNumNodes( gasnet_nodes() );
   _net->setNodeNum( gasnet_mynode() );

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
    unsigned int idx;
    unsigned int nodes = gasnet_nodes();
    void *segmentAddr[ nodes ];
    std::size_t segmentLen[ nodes ];
    void *pinnedSegmentAddr[ nodes ];
    std::size_t pinnedSegmentLen[ nodes ];
    
    gasnet_seginfo_t seginfoTable[ nodes ];
    gasnet_getSegmentInfo( seginfoTable, nodes );
    for ( idx = 0; idx < nodes; idx += 1)
    {
       pinnedSegmentAddr[ idx ] = seginfoTable[ idx ].addr;
       pinnedSegmentLen[ idx ] = seginfoTable[ idx ].size;
    }
    ClusterInfo::addPinnedSegments( nodes, pinnedSegmentAddr, pinnedSegmentLen );

    uintptr_t offset = pinnedSegmentLen[ gasnet_mynode() ] / 2;
    _packSegment = NEW SimpleAllocator( ( ( uintptr_t ) pinnedSegmentAddr[ gasnet_mynode() ] ) + offset , pinnedSegmentLen[ gasnet_mynode() ] / 2 );

   if ( _net->getNodeNum() == 0)
   {
    message0( "GasNet segment information:" );
    for ( idx = 0; idx < nodes; idx += 1)
    {
       message0( "\t"<< idx << "addr="<< pinnedSegmentAddr[idx]<<" len="<< pinnedSegmentLen[ idx ]  );
    }
      _pinnedAllocators.reserve( nodes );
      _pinnedAllocatorsLocks.reserve( nodes );
      _seqN = NEW Atomic<unsigned int>[nodes];
      
      _net->mallocSlaves( &segmentAddr[ 1 ], ClusterInfo::getNodeMem() );
      segmentAddr[ 0 ] = NULL;

      for ( idx = 0; idx < nodes; idx += 1)
      {
         segmentLen[ idx ] = ( idx == 0 ) ? 0 : ClusterInfo::getNodeMem(); 
         _pinnedAllocators[idx] = NEW SimpleAllocator( ( uintptr_t ) pinnedSegmentAddr[ idx ], pinnedSegmentLen[ idx ] / 2 );
         _pinnedAllocatorsLocks[idx] =  NEW Lock( );
         new (&_seqN[idx]) Atomic<unsigned int >( 0 );
      }
      _thisNodeSegment = _pinnedAllocators[0];
      ClusterInfo::addSegments( nodes, segmentAddr, segmentLen );
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
   gasnet_exit(0);
}

void GASNetAPI::poll ()
{
   if (myThread != NULL)
   {
      gasnet_AMPoll();
      checkForPutReqs();
      checkForFreeBufferReqs();
      checkWorkDoneReqs();
      //checkDeferredWorkReqs();
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

void GASNetAPI::sendWorkMsg ( unsigned int dest, void ( *work ) ( void * ), unsigned int dataSize, unsigned int wdId, unsigned int numPe, std::size_t argSize, char * arg, void ( *xlate ) ( void *, void * ), int arch, void *remoteWdAddr/*, void *remoteThd*/, std::size_t expectedData )
{
   std::size_t sent = 0;
   unsigned int msgCount = 0;

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
   //std::size_t expectedData = _sentWdData.getSentData( wdId );

   //message("To node " << dest << " wd id " << wdId << " seq " << (_seqN[dest].value() + 1) << " expectedData=" << expectedData);
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( wdId ) ; )
   NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_AM_WORK, id, 0, 0, dest ); )

      if (gasnet_AMRequestMedium13( dest, 205, &arg[ sent ], argSize - sent,
               ARG_LO( work ),
               ARG_HI( work ),
               ARG_LO( xlate ),
               ARG_HI( xlate ),
               ARG_LO( remoteWdAddr ),
               ARG_HI( remoteWdAddr ),
               ARG_LO( expectedData ),
               ARG_HI( expectedData ),
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

void GASNetAPI::_put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, std::size_t size, void *remoteTmpBuffer, unsigned int wdId, WD const &wd )
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
   }.
   else
#endif
   {
      if ( sys.getNetwork()->doIHaveToCheckForDataInOtherAddressSpaces() && gasnet_mynode() > 0) {
         getDataFromDevice( (uint64_t) localAddr, size );
      }
      while ( sent < size )
      {
         thisReqSize = ( ( size - sent ) <= gasnet_AMMaxLongRequest() ) ? size - sent : gasnet_AMMaxLongRequest();

         NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
         NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
         NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )

         if ( remoteTmpBuffer != NULL )
         { 
         NANOS_INSTRUMENT ( nanos_event_value_t xferSize = thisReqSize; )
         NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( ((uint64_t)remoteTmpBuffer) + sent ) ; )
         NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_PUT, id, sizeKey, xferSize, remoteNode ); )
            //fprintf(stderr, "try to send [%d:%p=>%d:%p,%ld < %f >].\n", gasnet_mynode(), (void*)localAddr, remoteNode, (void*)remoteAddr, size, *((double *)localAddr));
            if ( gasnet_AMRequestLong11( remoteNode, 210,
                     &( ( char *) localAddr )[ sent ],
                     thisReqSize,
                     ( char *) ( ( (char *) remoteTmpBuffer ) + sent ),
                     ARG_LO( ( ( uintptr_t ) ( ( uintptr_t ) remoteAddr ) + sent )),
                     ARG_HI( ( ( uintptr_t ) ( ( uintptr_t ) remoteAddr ) + sent )),
                     ARG_LO( ( ( uintptr_t ) remoteAddr ) ),
                     ARG_HI( ( ( uintptr_t ) remoteAddr ) ),
                     ARG_LO( size ),
                     ARG_HI( size ),
                     wdId,
                     ARG_LO( &wd ),
                     ARG_HI( &wd ),
                     ( sent == 0 ),
                     ( ( sent + thisReqSize ) == size )
                     ) != GASNET_OK)
            {
               fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
            }
            //fprintf(stderr, "Req sent to node %d.\n", remoteNode);
         }
         else { fprintf(stderr, "error sending a PUT to node %d, did not get a tmpBuffer\n", remoteNode ); }
         sent += thisReqSize;
      }
   }
}

void GASNetAPI::_putStrided1D ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, void *localPack, std::size_t size, std::size_t count, std::size_t ld, void *remoteTmpBuffer, unsigned int wdId, WD const &wd )
{
   std::size_t sent = 0, thisReqSize;
   std::size_t realSize = size * count;
   txBytes += realSize;
   _totalBytes += realSize;
   {
      if ( sys.getNetwork()->doIHaveToCheckForDataInOtherAddressSpaces() && gasnet_mynode() > 0) {
         std::cerr << "ERROR, not supported yet" << std::endl;
         getDataFromDevice( (uint64_t) localAddr, size );
      }
      while ( sent < realSize )
      {
         thisReqSize = ( ( realSize - sent ) <= gasnet_AMMaxLongRequest() ) ? realSize - sent : gasnet_AMMaxLongRequest();

         NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
         NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
         NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )

         if ( remoteTmpBuffer != NULL )
         { 
            NANOS_INSTRUMENT ( nanos_event_value_t xferSize = thisReqSize; )
            NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( ((uint64_t)remoteTmpBuffer) + sent ) ; )
            NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_PUT, id, sizeKey, xferSize, remoteNode ); )
            if ( gasnet_AMRequestLong11( remoteNode, 220,
                     &( ( char *) localPack )[ sent ],
                     thisReqSize,
                     ( char *) ( ( (char *) remoteTmpBuffer ) + sent ),
                     ARG_LO( ( ( uintptr_t ) remoteAddr ) ),
                     ARG_HI( ( ( uintptr_t ) remoteAddr ) ),
                     ARG_LO( size ),
                     ARG_HI( size ),
                     count,
                     ld,
                     wdId,
                     ARG_LO( &wd ),
                     ARG_HI( &wd ),
                     ( sent == 0 ),
                     ( ( sent + thisReqSize ) == realSize )
                     ) != GASNET_OK)
            {
               fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
            }
         }
         else { std::cerr <<"Unsupported. this node is " <<gasnet_mynode()<< std::endl; }
         sent += thisReqSize;
      }
   }
}

void GASNetAPI::putStrided1D ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, void *localPack, std::size_t size, std::size_t count, std::size_t ld, unsigned int wdId, WD const &wd ) {
   if ( gasnet_mynode() != 0 ) fatal0("Error, cant use ::put from node != than 0"); 
   void *tmp = NULL;
   while( tmp == NULL ) {
      _pinnedAllocatorsLocks[ remoteNode ]->acquire();
      tmp = _pinnedAllocators[ remoteNode ]->allocate( size * count );
      _pinnedAllocatorsLocks[ remoteNode ]->release();
      if ( tmp == NULL ) _net->poll(0);
   }
   if ( tmp == NULL ) std::cerr << "what... "<< tmp << std::endl; 
   _putStrided1D( remoteNode, remoteAddr, localAddr, localPack, size, count, ld, tmp, wdId, wd );
}

void GASNetAPI::put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, std::size_t size, unsigned int wdId, WD const &wd )
{
   if ( gasnet_mynode() != 0 ) fatal0("Error, cant use ::put from node != than 0"); 
   void *tmp = NULL;
   while( tmp == NULL ) {
      _pinnedAllocatorsLocks[ remoteNode ]->acquire();
      tmp = _pinnedAllocators[ remoteNode ]->allocate( size );
      _pinnedAllocatorsLocks[ remoteNode ]->release();
      if ( tmp == NULL ) _net->poll(0);
   }
   _put( remoteNode, remoteAddr, localAddr, size, tmp, wdId, wd );
}

//Lock getLock;
#ifndef GASNET_SEGMENT_EVERYTHING
Lock getLockGlobal;
#endif

void GASNetAPI::get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, std::size_t size, volatile int *requestComplete )
{
   std::size_t sent = 0, thisReqSize;
   void *addr;

   addr = localAddr;

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
      if ( gasnet_AMRequestShort12( remoteNode, 211,
               ARG_LO( ( ( uintptr_t ) ( ( uintptr_t ) addr ) + sent )  ),
               ARG_HI( ( ( uintptr_t ) ( ( uintptr_t ) addr ) + sent )  ),
               ARG_LO( remoteAddr + sent ),
               ARG_HI( remoteAddr + sent ),
               ARG_LO( remoteAddr ),
               ARG_HI( remoteAddr ),
               ARG_LO( thisReqSize ),
               ARG_HI( thisReqSize ),
               ARG_LO( size ),
               ARG_HI( size ),
               ARG_LO( (uintptr_t) (( ( sent + thisReqSize ) == size ) ? requestComplete : NULL )),
               ARG_HI( (uintptr_t) (( ( sent + thisReqSize ) == size ) ? requestComplete : NULL )) ) != GASNET_OK)
      {
         fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
      }
      sent += thisReqSize;
   }

   rxBytes += size;
   _totalBytes += size;

#ifndef GASNET_SEGMENT_EVERYTHING
   // copy the data to the correct addr;
   //std::cerr << "GET: copy data from buffer to real addr: " << (void *) localAddr << " localbuff " << addr << " size used is " << size << std::endl;
   //{
   //int *a = (int *) addr;
   //for (unsigned int i = 0; i < (size / sizeof(int)); i += 1, a += 1) std::cerr << *a << " ";
   //std::cerr << std::endl;
   //}
   // this is done in GetRequest.clear() XXX ::memcpy( localAddr, addr, size );
   // this is done in GetRequest.clear() XXX getLockGlobal.acquire();
   // this is done in GetRequest.clear() XXX _thisNodeSegment->free( addr );
   // this is done in GetRequest.clear() XXX getLockGlobal.release();
   //fprintf(stderr, "get data is %f, addr %p\n", *((float *)localAddr), localAddr );
#endif
}

std::size_t GASNetAPI::getMaxGetStridedLen() const {
   return ( std::size_t ) gasnet_AMMaxLongReply();
}

void GASNetAPI::getStrided1D ( void *packedAddr, unsigned int remoteNode, uint64_t remoteTag, uint64_t remoteAddr, std::size_t size, std::size_t count, std::size_t ld, volatile int *requestComplete )
{
   std::size_t thisReqSize = size * count;
   void *addr = packedAddr;

   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) requestComplete ; )
   NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent ( NANOS_XFER_GET, id, sizeKey, xferSize, remoteNode ); )

   //fprintf(stderr, "n:%d send get req to node %d(src=%p, dst=%p localtag=%p, size=%ld)\n", gasnet_mynode(), remoteNode, (void *) remoteAddr, (void *) ( ( uintptr_t ) ( ( uintptr_t ) addr ) + sent ), localAddr, thisReqSize  );
   if ( gasnet_AMRequestShort12( remoteNode, 221,
            ARG_LO( ( ( uintptr_t ) addr ) ),
            ARG_HI( ( ( uintptr_t ) addr ) ),
            ARG_LO( remoteAddr ),
            ARG_HI( remoteAddr ),
            ARG_LO( remoteTag ),
            ARG_HI( remoteTag ),
            ARG_LO( size ),
            ARG_HI( size ),
            count, ld,
            ARG_LO( (uintptr_t) ( requestComplete ) ),
            ARG_HI( (uintptr_t) ( requestComplete ) ) ) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
   }

   rxBytes += thisReqSize;
   _totalBytes += thisReqSize;

   //if ( sys.getNetwork()->doIHaveToCheckForDataInOtherAddressSpaces() && gasnet_mynode() > 0)
   //{
   //   invalidateDataFromDevice( (uint64_t) localAddr, size );
   //}
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

   if ( gasnet_AMRequestMedium0( dest, 209, ( void * ) masterHostname, ::strlen( masterHostname ) ) != GASNET_OK )
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest );
   }
}

void GASNetAPI::sendRequestPut( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, std::size_t len, unsigned int wdId, WD const &wd )
{
   _totalBytes += len;
   sendWaitForRequestPut( dataDest, dstAddr );
   void *tmpBuffer = NULL;
   while ( tmpBuffer == NULL ) {
      _pinnedAllocatorsLocks[ dataDest ]->acquire();
      tmpBuffer = _pinnedAllocators[ dataDest ]->allocate( len );
      _pinnedAllocatorsLocks[ dataDest ]->release();
      if ( tmpBuffer == NULL ) _net->poll(0);
   }

   if ( gasnet_AMRequestShort12( dest, 214,
       ARG_LO( dstAddr ), ARG_HI( dstAddr ),
       ARG_LO( ( ( uintptr_t ) origAddr ) ), ARG_HI( ( ( uintptr_t ) origAddr ) ),
       ARG_LO( ( ( uintptr_t ) tmpBuffer ) ), ARG_HI( ( ( uintptr_t ) tmpBuffer ) ),
	    ARG_LO( len ),
	    ARG_HI( len ),
	    wdId,
	    ARG_LO( &wd ),
	    ARG_HI( &wd ),
       dataDest ) != GASNET_OK )

   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GASNetAPI::sendRequestPutStrided1D( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, std::size_t len, std::size_t count, std::size_t ld, unsigned int wdId, WD const &wd )
{
   _totalBytes += ( len * count );
   NANOS_INSTRUMENT( InstrumentState inst0(NANOS_SEND_WAIT_FOR_REQ_PUT); );
   sendWaitForRequestPut( dataDest, dstAddr );
   NANOS_INSTRUMENT( inst0.close(); );
   NANOS_INSTRUMENT( InstrumentState inst1(NANOS_GET_PINNED_ADDR); );

   void *tmpBuffer = NULL;
   while ( tmpBuffer == NULL ) {
      _pinnedAllocatorsLocks[ dataDest ]->acquire();
      tmpBuffer = _pinnedAllocators[ dataDest ]->allocate( len * count );
      _pinnedAllocatorsLocks[ dataDest ]->release();
      if ( tmpBuffer == NULL ) _net->poll(0);
   }
   NANOS_INSTRUMENT( inst1.close(); );

   NANOS_INSTRUMENT( InstrumentState inst2(NANOS_SEND_PUT_REQ); );

   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( ((uint64_t)tmpBuffer) ) ; )
   NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_PUT, id, sizeKey, xferSize, dest ); )
   if ( gasnet_AMRequestShort14( dest, 222,
       ARG_LO( dstAddr ), ARG_HI( dstAddr ),
       ARG_LO( ( ( uintptr_t ) origAddr ) ), ARG_HI( ( ( uintptr_t ) origAddr ) ),
       ARG_LO( ( ( uintptr_t ) tmpBuffer ) ), ARG_HI( ( ( uintptr_t ) tmpBuffer ) ),
	    ARG_LO( len ),
	    ARG_HI( len ),
       ( gasnet_handlerarg_t ) count,
       ( gasnet_handlerarg_t ) ld,
	    wdId,
	    ARG_LO( &wd ),
	    ARG_HI( &wd ),
       dataDest ) != GASNET_OK )

   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
   NANOS_INSTRUMENT( inst2.close(); );
}

void GASNetAPI::setNewMasterDirectory(NewRegionDirectory *dir)
{
   _newMasterDir = dir;
}

void GASNetAPI::sendWaitForRequestPut( unsigned int dest, uint64_t addr )
{
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( addr ) ; )
   NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_WAIT_REQ_PUT, id, sizeKey, xferSize, dest ); )
   if ( gasnet_AMRequestShort2( dest, 218,
            ARG_LO( addr ), ARG_HI( addr ) ) != GASNET_OK )
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GASNetAPI::sendFreeTmpBuffer( void *addr, WD const *wd )
{
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( addr ) ; )
   NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_FREE_TMP_BUFF, id, sizeKey, xferSize, 0 ); )
   if ( gasnet_AMRequestShort4( 0, 219,
            ARG_LO( addr ), ARG_HI( addr ), ARG_LO( wd ), ARG_HI( wd ) ) != GASNET_OK )
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

void GASNetAPI::getDataFromDevice( uint64_t addr, std::size_t len ) {
   NewDirectory::LocationInfoList locations;
   unsigned int currentVersion;
   nanos_region_dimension_internal_t aDimension = { len, 0, len };
   CopyData cd( addr, NANOS_SHARED, true, true, 1, &aDimension, 0);
   Region reg = NewRegionDirectory::build_region( cd );
   _newMasterDir->lock();
   _newMasterDir->masterGetLocation( reg, locations, currentVersion ); 
   _newMasterDir->addAccess( reg, 0, currentVersion ); 
   _newMasterDir->unlock();
   for ( NewDirectory::LocationInfoList::iterator it = locations.begin(); it != locations.end(); it++ ) {
      if (!it->second.isLocatedIn( 0 ) ) { 
         unsigned int loc = it->second.getFirstLocation();
         //std::cerr << "Houston, we have a problem, data is not in Host and we need it back. HostAddr: " << (void *) (((it->first)).getFirstValue()) << it->second << std::endl;
         sys.getCaches()[ loc ]->syncRegion( it->first /*, it->second.getAddressOfLocation( loc )*/ );
         //std::cerr <<"[" << gasnet_mynode() << "] Sync data to host mem, got " << *((double *) it->first.getFirstValue())<< " addr is " << (void *) it->first.getFirstValue() << std::endl;
      }
     //else { /*if ( sys.getNetwork()->getNodeNum() == 0)*/ std::cerr << "["<<sys.getNetwork()->getNodeNum()<<"]  All ok, checked directory " << _newMasterDir  <<" location is " << it->second << std::endl; }
   }
}

void GASNetAPI::invalidateDataFromDevice( uint64_t addr, std::size_t len ) {
   NewDirectory::LocationInfoList locations;
   unsigned int currentVersion;
   nanos_region_dimension_internal_t aDimension = { len, 0, len };
   CopyData cd( addr, NANOS_SHARED, true, true, 1, &aDimension, 0);
   Region reg = NewRegionDirectory::build_region( cd );
   _newMasterDir->lock();
   //_newMasterDir->masterRegisterAccess( reg, true, true /* will increase version number */, 0, addr /*this is currently unused */, locations );
   _newMasterDir->masterGetLocation( reg, locations, currentVersion ); 
   _newMasterDir->addAccess( reg, 0, currentVersion + 1 ); 
   _newMasterDir->unlock();
}

#if 0
void GASNetAPI::ReceivedWDData::addData( unsigned int wdId, std::size_t size ) {
   _lock.acquire();
   struct recvDataInfo &info = _recvWdData[ wdId ];
   info._count += size;
   if ( info._wd != NULL && info._expected == info._count ) {
      WD *wd = info._wd;
      if ( _recvWdData.erase( wdId ) != 1) std::cerr <<"Error removing from map: "<<__FUNCTION__<< " @ " << __FILE__<<":"<<__LINE__<< std::endl;
      _lock.release();
      //release wd
      sys.submit( *wd );
      _recvSeqN++;
      //std::cerr <<"["<< gasnet_mynode()<< "] release wd (by data) new seq is " << _recvSeqN.value()   << std::endl;
   } else {
      _lock.release();
   }
}
void GASNetAPI::ReceivedWDData::addWD( unsigned int wdId, WorkDescriptor *wd, std::size_t expectedData ) {
   _lock.acquire();
   struct recvDataInfo &info = _recvWdData[ wdId ];
   info._wd = wd;
   info._expected = expectedData;
   //std::cerr <<"["<< gasnet_mynode()<< "] addWD with expected data: " << expectedData << " current count: " << info._count  << std::endl;
   if ( info._expected == info._count ) {
      if ( _recvWdData.erase( wdId ) != 1) std::cerr <<"Error removing from map: "<<__FUNCTION__<< " @ " << __FILE__<<":"<<__LINE__<< std::endl;
      _lock.release();
      //release wd
      sys.submit( *wd );
      _recvSeqN++;
   //std::cerr <<"["<< gasnet_mynode()<< "] release wd (by wd) new seq is " << _recvSeqN.value()   << std::endl;
   } else {
      _lock.release();
   }
}

void GASNetAPI::SentWDData::addSentData( unsigned int wdId, std::size_t sentData ) {
   _lock.acquire();
   _sentWdData[ wdId ] += sentData;//assumes that if no data was yet sent, the elem is initialized to 0
   _lock.release();
}

std::size_t GASNetAPI::SentWDData::getSentData( unsigned int wdId ) {
   _lock.acquire();
   std::size_t wdSentData = _sentWdData[ wdId ];
   if ( _sentWdData.erase( wdId ) != 1) std::cerr <<"Error removing from map: "<<__FUNCTION__<< " @ " << __FILE__<<":"<<__LINE__<< std::endl;
   _lock.release();
   return wdSentData;
}
#endif

SimpleAllocator *GASNetAPI::getPackSegment() const {
   return _packSegment;
}

void *GASNetAPI::allocateReceiveMemory( std::size_t len ) {
   void *addr = NULL;
   do {
      getLockGlobal.acquire();
      addr = _thisNodeSegment->allocate( len );
      getLockGlobal.release();
      if ( addr == NULL ) _net->poll(0);
   } while (addr == NULL);
   return addr;
}

void GASNetAPI::freeReceiveMemory( void * addr ) {
   getLockGlobal.acquire();
   _thisNodeSegment->free( addr );
   getLockGlobal.release();
}
