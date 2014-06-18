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
#include "clusterplugin_decl.hpp"
#include "smpdd.hpp"

#ifdef GPU_DEV
//FIXME: GPU Support
#include "gpudd.hpp"
#include "gpudevice_decl.hpp"
#endif

#include "system.hpp"
#include "os.hpp"
#include "clusterdevice_decl.hpp"
#include "instrumentation.hpp"
#include "osallocator_decl.hpp"
#include "requestqueue.hpp"
#include "atomic.hpp"
#include <cstddef>

//#define HALF_PRESEND

#ifdef HALF_PRESEND
Atomic<int> wdindc = 0;
WD* buffWD = NULL;
#endif

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

#define _emitPtPEvents 1

GASNetAPI *GASNetAPI::_instance = 0;

GASNetAPI *GASNetAPI::getInstance() {
   return _instance;
}

GASNetAPI::GASNetAPI( ClusterPlugin &p ) : _plugin( p ), _net( 0 ), _rwgGPU( 0 ), _rwgSMP( 0 ), _packSegment( 0 ),
   _pinnedAllocators(), _pinnedAllocatorsLocks(),
   _seqN( 0 ), _dataSendRequests(), _freeBufferReqs(), _workDoneReqs(), _rxBytes( 0 ), _txBytes( 0 ), _totalBytes( 0 ),
   _numSegments( 0 ), _segmentAddrList( NULL ), _segmentLenList ( NULL ) {
   _instance = this;
}

GASNetAPI::~GASNetAPI(){
}

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

void GASNetAPI::print_copies( WD const *wd, int deps )
{
#if 1
   unsigned int i;
   fprintf(stderr, "node %d submit slave %s wd %d with %d deps, copies are: ", gasnet_mynode(), /*(((WG*) wd)->getParent() == (WG*) GASNetAPI::_rwgGPU ? "GPU" : "SMP")*/"n/a", wd->getHostId(), deps );
   for ( i = 0; i < wd->getNumCopies(); i++)
      fprintf(stderr, "%s%s:%p ", ( wd->getCopies()[i].isInput() ? "r" : "-" ), ( wd->getCopies()[i].isOutput() ? "w" : "-" ), (void *) wd->getCopies()[i].getAddress() );
   fprintf(stderr, "\n");
#endif
}



GASNetAPI::SendDataPutRequestPayload::SendDataPutRequestPayload( unsigned int seqNumber, void *origAddr, void *dstAddr, std::size_t len,
   std::size_t count, std::size_t ld, unsigned int dest, unsigned int wdId, void *tmpBuffer, WD const *wd, Functor *func,
   void *hostObject, reg_t hostRegId ) :
   _seqNumber( seqNumber ), _origAddr( origAddr ), _destAddr( dstAddr ), _len( len ), _count( count ), _ld( ld ), _destination( dest ),
   _wdId( wdId ), _tmpBuffer( tmpBuffer ), _wd( wd ), _functor( func ), _hostObject( hostObject ), _hostRegId( hostRegId ) {
}

GASNetAPI::SendDataGetRequestPayload::SendDataGetRequestPayload( unsigned int seqNumber, void *origAddr, void *dstAddr, std::size_t len,
   std::size_t count, std::size_t ld, void *waitObj, CopyData const &cd ) :
   _seqNumber( seqNumber ), _origAddr( origAddr ), _destAddr( dstAddr ), _len( len ), _count( count ), _ld( ld ), _waitObj( waitObj ),
   _cd( cd ) {
}


GASNetAPI::GASNetSendDataRequest::GASNetSendDataRequest( GASNetAPI *api, unsigned int seqNumber, void *origAddr, void *destAddr, std::size_t len, std::size_t count, std::size_t ld, unsigned int dst, unsigned int wdId, void *hostObject, reg_t hostRegId ) :
   SendDataRequest( api, seqNumber, origAddr, destAddr, len, count, ld, dst, wdId, hostObject, hostRegId ), _gasnetApi( api ) {
}

GASNetAPI::SendDataPutRequest::SendDataPutRequest( GASNetAPI *api, SendDataPutRequestPayload *msg ) :
   GASNetSendDataRequest( api, msg->_seqNumber, msg->_origAddr, msg->_destAddr, msg->_len, msg->_count, msg->_ld, msg->_destination, msg->_wdId, msg->_hostObject, msg->_hostRegId ), _tmpBuffer( msg->_tmpBuffer ), _wd( msg->_wd ), _functor( msg->_functor ) {
}

GASNetAPI::SendDataPutRequest::~SendDataPutRequest() {
}

void GASNetAPI::SendDataPutRequest::doSingleChunk() {
    //std::cerr << "process request for wd " << _wdId << " to node " << getDestination() << " dstAddr is " << (void *) _destAddr << std::endl;
   _gasnetApi->_put( getDestination(), (uint64_t) _destAddr, _origAddr, _len, _tmpBuffer, _wdId, *_wd, _functor, _hostObject, _hostRegId );
}

void GASNetAPI::SendDataPutRequest::doStrided( void *localAddr ) {
    //std::cerr << "process strided request for wd " << _wdId << " to node " << getDestination() << " dstAddr is " << (void *) _destAddr << std::endl;
   _gasnetApi->_putStrided1D( getDestination(), (uint64_t) _destAddr, _origAddr, localAddr, _len, _count, _ld, _tmpBuffer, _wdId, *_wd, _functor, _hostObject, _hostRegId );
}

GASNetAPI::SendDataGetRequest::SendDataGetRequest( GASNetAPI *api, unsigned int seqNumber, void *origAddr, void *destAddr, std::size_t len, std::size_t count, std::size_t ld, void *waitObj, CopyData const &cd, nanos_region_dimension_internal_t *dims ) :
   GASNetSendDataRequest( api, seqNumber, origAddr, destAddr, len, count, ld, 0, 0, (void *) cd.getHostBaseAddress(),
   cd.getHostRegionId() ), _waitObj( waitObj ), _cd( cd ) {
   nanos_region_dimension_internal_t *cd_dims = NEW nanos_region_dimension_internal_t[ _cd.getNumDimensions() ];
   ::memcpy( cd_dims, dims, sizeof(nanos_region_dimension_internal_t) * _cd.getNumDimensions());
   _cd.setDimensions( cd_dims );
}

GASNetAPI::SendDataGetRequest::~SendDataGetRequest() {
   delete[] _cd.getDimensions();
}

void GASNetAPI::SendDataGetRequest::doSingleChunk() {
   std::size_t sent = 0, thisReqSize;
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   while ( sent < _len )
   {
      thisReqSize = ( ( _len - sent ) <= gasnet_AMMaxLongRequest() ) ? _len - sent : gasnet_AMMaxLongRequest();

      if ( _emitPtPEvents ) {
         NANOS_INSTRUMENT ( nanos_event_value_t xferSize = thisReqSize; )
         NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( ((uint64_t)_destAddr) + sent ) ; )
         NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_DATA, id, sizeKey, xferSize, 0 ); )
      }

      if ( gasnet_AMRequestLong2( 0, 212,
               &( ( char *) _origAddr )[ sent ],
               thisReqSize,
               ( char *) ( ( (char *) _destAddr ) + sent ),
               ( ( sent + thisReqSize ) == _len ) ? ARG_LO( _waitObj ) : 0,
               ( ( sent + thisReqSize ) == _len ) ? ARG_HI( _waitObj ) : 0
               ) != GASNET_OK )
      {
         fprintf(stderr, "gasnet: Error sending a message to node %d.\n", 0);
      }
      sent += thisReqSize;
   }
}

void GASNetAPI::SendDataGetRequest::doStrided( void *localAddr ) {
   if ( gasnet_AMRequestLong2( 0, 212, localAddr, _len*_count, _destAddr, ARG_LO( _waitObj ), ARG_HI( _waitObj ) ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error sending reply msg.\n" );
   }
}

void GASNetAPI::processSendDataRequest( SendDataRequest *req ) {
   _dataSendRequests.add( req );
}

void GASNetAPI::checkForPutReqs()
{
   SendDataRequest *req = _dataSendRequests.tryFetch();
   if ( req != NULL ) {
      req->doSend();
      delete req;
   }
}

void GASNetAPI::enqueueFreeBufferNotify( void *tmpBuffer, WD const *wd, Functor *f )
{
   FreeBufferRequest *addrWd = NEW FreeBufferRequest( tmpBuffer, wd, f );
   _freeBufferReqs.add( addrWd );
}

void GASNetAPI::checkForFreeBufferReqs()
{
   FreeBufferRequest *req = _freeBufferReqs.tryFetch();
   if ( req != NULL ) {
      sendFreeTmpBuffer( req->address, req->wd, req->functor );
      delete req;
   }
}

void GASNetAPI::checkWorkDoneReqs()
{
   std::pair<void *, unsigned int> *rwd = _workDoneReqs.tryFetch();
   if ( rwd != NULL ) {
      _sendWorkDoneMsg( 0, rwd->first, rwd->second );
      delete rwd;
   }
}

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
   WorkDescriptor *rwg;

   if (gasnet_AMGetMsgSource(token, &src_node) != GASNET_OK)
   {
      fprintf(stderr, "gasnet: Error obtaining node information.\n");
   }
   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( rmwd ) ; )
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

      if (getInstance()->_rwgSMP == NULL) 
         getInstance()->_rwgSMP = getInstance()->_plugin.getRemoteWorkDescriptor( 0 );

      rwg = (WorkDescriptor *) getInstance()->_rwgSMP;
   }
#ifdef GPU_DEV
   else if (arch == 1)
   {
      //FIXME: GPU support
      devPtr = &newDeviceGPU;

      if (getInstance()->_rwgGPU == NULL)
         getInstance()->_rwgGPU = getInstance()->_plugin.getRemoteWorkDescriptor( 1 );

      rwg = (WorkDescriptor *) getInstance()->_rwgGPU;
   }
#endif
   else
   {
      rwg = NULL;
      fprintf(stderr, "Unsupported architecture\n");
   }

   sys.createWD( &localWD, (std::size_t) 1, devPtr, (std::size_t) dataSize, (int) ( sizeof(void *) ), (void **) &data, (WD *)rwg, (nanos_wd_props_t *) NULL, (nanos_wd_dyn_props_t *) NULL, (std::size_t) numCopies, newCopiesPtr, num_dimensions, dimensions_ptr, xlate, NULL, NULL );

   VERBOSE_AM( std::cerr << __FUNCTION__ << " from " << src_node << " wid "<< wdId << " wdAddr "<< (void *) localWD << std::endl; );
   std::memcpy(data, work_data, dataSize);

   //unsigned int numDeps = *( ( int * ) &work_data[ dataSize + sizeof( int ) + numCopies * sizeof( CopyData ) + sizeof( int ) + num_dimensions * sizeof( nanos_region_dimension_t ) ] );
   //uint64_t *depTags = ( ( uint64_t * ) &work_data[ dataSize + sizeof( int ) + numCopies * sizeof( CopyData ) + sizeof( int ) + num_dimensions * sizeof( nanos_region_dimension_t ) + sizeof( int ) ] );

   // Set copies and dimensions, getDimensions() returns an index here, instead of a pointer,
   // the index is the position inside the dimension array that must be set as the base address for the dimensions
   CopyData *recvCopies = ( ( CopyData *) &work_data[ dataSize + sizeof( int ) ] );
   nanos_region_dimension_t *recvDimensions = ( ( nanos_region_dimension_t * ) &work_data[ dataSize + sizeof( int ) + numCopies * sizeof( CopyData ) + sizeof(int) ] );
   if ( numCopies > 0 ) {
      memcpy( *dimensions_ptr, recvDimensions, num_dimensions * sizeof(nanos_region_dimension_t) );
   }
   for (i = 0; i < numCopies; i += 1)
   {
      new ( &newCopies[i] ) CopyData( recvCopies[i] );
      newCopies[i].setDimensions( (*dimensions_ptr) + ( ( uintptr_t ) recvCopies[i].getDimensions() ) );
      //fprintf(stderr, "Copy %d has HostAddr 0x%lx, BaseAddr 0x%p\n", i, newCopies[i].getHostBaseAddress(), newCopies[i].getBaseAddress() );
   }

   localWD->setHostId( wdId );
   localWD->setRemoteAddr( rmwd );

   getInstance()->_net->notifyWork(expectedData, localWD, seq);

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
   VERBOSE_AM( std::cerr << __FUNCTION__ << " host wd addr "<< (void *) addr << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( addr ) ; )
      NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_AM_WORK_DONE, id, 0, 0, src_node ); )
   }

   sys.getNetwork()->notifyWorkDone( src_node, addr, peId );
   VERBOSE_AM( std::cerr << __FUNCTION__ << " from "<< src_node <<" host wd addr "<< (void *) addr <<" done." << std::endl; );
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
      gasnet_handlerarg_t lastMsg, 
      gasnet_handlerarg_t functorLo,
      gasnet_handlerarg_t functorHi,
      gasnet_handlerarg_t hostObjectLo, 
      gasnet_handlerarg_t hostObjectHi, 
      gasnet_handlerarg_t hostRegId ) 
{
   gasnet_node_t src_node;
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   void *realAddr = ( int * ) MERGE_ARG( realAddrHi, realAddrLo );
   void *realTag = ( int * ) MERGE_ARG( realTagHi, realTagLo );
   Functor *f = ( Functor * ) MERGE_ARG( functorHi, functorLo );
   WD *wd = ( WD * ) MERGE_ARG( wdHi, wdLo );
   std::size_t totalLen = ( std::size_t ) MERGE_ARG( totalLenHi, totalLenLo );
   VERBOSE_AM( std::cerr << gasnet_mynode() << ": " << __FUNCTION__ << " from " << src_node << " size " << len << " addrTag " << (void*) realTag << " addrAddr "<< (void*)realAddr<< std::endl; );

   getInstance()->_rxBytes += len;

   
   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = len; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( buf ) ; )
      NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_DATA, id, sizeKey, xferSize, src_node ); )
   }

   if ( realAddr != NULL )
   {
      ::memcpy( realAddr, buf, len );
   }
   if ( lastMsg )
   {
      uintptr_t localAddr =  ( ( uintptr_t ) buf ) - ( ( uintptr_t ) realAddr - ( uintptr_t ) realTag );
      //char* realAddrPtr = (char *) realTag;
      //char* localAddrPtr = ( (char *) ( ( ( uintptr_t ) buf ) + ( ( uintptr_t ) len ) - ( uintptr_t ) totalLen ) );
      //fprintf(stderr, "[%d] amPut from %d: dst[%p]=%f buff[%p]=%f\n", gasnet_mynode(), src_node, &realAddrPtr[ 0 ], *((double*)&realAddrPtr[0]), &localAddrPtr[0], *((double*)&localAddrPtr[0]) );
      getInstance()->enqueueFreeBufferNotify( ( void * ) localAddr, wd, f );
      void *hostObject = ( void * ) MERGE_ARG( hostObjectHi, hostObjectLo );
      getInstance()->_net->notifyPut( src_node, wdId, totalLen, 1, 0, (uint64_t) realTag, hostObject, hostRegId );
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
      gasnet_handlerarg_t lastMsg, 
      gasnet_handlerarg_t functorLo,
      gasnet_handlerarg_t functorHi,
      gasnet_handlerarg_t hostObjectLo, 
      gasnet_handlerarg_t hostObjectHi, 
      gasnet_handlerarg_t hostRegId ) 
{
   gasnet_node_t src_node;
   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   void *realTag = ( int * ) MERGE_ARG( realTagHi, realTagLo );
   std::size_t size = ( std::size_t ) MERGE_ARG( sizeHi, sizeLo );
   Functor *f = ( Functor * ) MERGE_ARG( functorHi, functorLo );
   std::size_t totalLen = size * count;
   WD *wd = ( WD * ) MERGE_ARG( wdHi, wdLo );

   getInstance()->_rxBytes += len;

   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = len; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( buf ) ; )
      NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_DATA, id, sizeKey, xferSize, src_node ); )
   }

   if ( lastMsg )
   {
      char* realAddrPtr = (char *) realTag;
      char* localAddrPtr = ( (char *) ( ( ( uintptr_t ) buf ) + ( ( uintptr_t ) len ) - ( uintptr_t ) totalLen ) );
      //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_STRIDED_COPY_UNPACK); );
      for ( int i = 0; i < count; i += 1 ) {
         ::memcpy( &realAddrPtr[ i * ld ], &localAddrPtr[ i * size ], size );
         //if (i == 0)fprintf(stderr, "[%d] amPutStrided1D from %d: dst[%p]=%f buff[%p]=%f\n", gasnet_mynode(), src_node, &realAddrPtr[ i * ld], *((double*)&realAddrPtr[i*ld]), &localAddrPtr[i*size], *((double*)&localAddrPtr[i*size]) );
      }
      //NANOS_INSTRUMENT( inst2.close(); );
      uintptr_t localAddr = ( ( uintptr_t ) buf ) + ( ( uintptr_t ) len ) - ( uintptr_t ) totalLen;
      getInstance()->enqueueFreeBufferNotify( ( void * ) localAddr, wd, f );
      void *hostObject = ( void * ) MERGE_ARG( hostObjectHi, hostObjectLo );
      getInstance()->_net->notifyPut( src_node, wdId, size, count, ld, (uint64_t)realTag, hostObject, hostRegId );
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
   GetRequest *waitObj = ( GetRequest * ) MERGE_ARG( waitObjHi, waitObjLo );

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   //fprintf(stderr, "get reply from %d: data is %d waitObj %p\n", src_node , *((int *)buf), waitObj);
   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = len; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) buf; )
      NANOS_INSTRUMENT ( instr->raiseClosePtPEvent ( NANOS_XFER_DATA, id, sizeKey, xferSize, src_node ); )
   }

   if ( waitObj != NULL )
   {
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
   GetRequest *waitObj = ( GetRequest * ) MERGE_ARG( waitObjHi, waitObjLo );

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
   NANOS_INSTRUMENT ( nanos_event_value_t xferSize = len; )
   NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) buf; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEvent ( NANOS_XFER_DATA, id, sizeKey, xferSize, src_node ); )

   if ( waitObj != NULL )
   {
      //   waitObj->clear();
      waitObj->complete();
   }
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amGet( gasnet_token_t token, void *buff, std::size_t nbytes ) {
   gasnet_node_t src_node;
   SendDataGetRequestPayload *msg = ( SendDataGetRequestPayload * ) buff;
   nanos_region_dimension_internal_t *dims = (nanos_region_dimension_internal_t *)( ((char *)buff)+ sizeof( SendDataGetRequestPayload ) );

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) msg->_waitObj; )
      NANOS_INSTRUMENT ( instr->raiseClosePtPEvent ( NANOS_XFER_REQ, id, sizeKey, xferSize, src_node ); )
   }

   getInstance()->_txBytes += msg->_len;

   SendDataGetRequest *req = NEW SendDataGetRequest( getInstance(), msg->_seqNumber, msg->_destAddr, msg->_origAddr, msg->_len, 1, 0, msg->_waitObj, msg->_cd, dims );
   getInstance()->_net->notifyRegionMetaData( &( req->_cd ) );
   getInstance()->_net->notifyGet( req );
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
   //NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_DATA, id, sizeKey, xferSize, src_node ); )
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amRequestPut( gasnet_token_t token, void *buff, std::size_t nbytes ) {
   gasnet_node_t src_node;
   SendDataPutRequestPayload *msg = ( SendDataPutRequestPayload * ) buff;

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   VERBOSE_AM( std::cerr << gasnet_mynode() << ": " << __FUNCTION__ << " from " << src_node << " to " << msg->_destination <<" size " << msg->_len << " origAddr " << (void*) msg->_origAddr << " destAddr "<< (void*)msg->_destAddr<< std::endl; );

   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) msg->_tmpBuffer; )
      NANOS_INSTRUMENT ( instr->raiseClosePtPEvent ( NANOS_XFER_REQ, id, sizeKey, xferSize, src_node ); )
   }

   SendDataPutRequest *req = NEW SendDataPutRequest( getInstance(), msg );
   getInstance()->_net->notifyRequestPut( req );
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amRequestPutStrided1D( gasnet_token_t token, void *buff, std::size_t nbytes ) {
   SendDataPutRequestPayload *msg = ( SendDataPutRequestPayload * ) buff;
   gasnet_node_t src_node;

   //NANOS_INSTRUMENT( InstrumentState inst(NANOS_amRequestPutStrided1D); );

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( msg->_tmpBuffer ) ; )
      NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_REQ, id, sizeKey, xferSize, src_node ); )
   }

   SendDataPutRequest *req = NEW SendDataPutRequest( getInstance(), msg );
   getInstance()->_net->notifyRequestPut( req );
   //NANOS_INSTRUMENT( inst.close(); );
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amWaitRequestPut( gasnet_token_t token, 
      gasnet_handlerarg_t addrLo,
      gasnet_handlerarg_t addrHi,
      gasnet_handlerarg_t wdId,
      gasnet_handlerarg_t seqNumber )
{
   void *addr = ( void * ) MERGE_ARG( addrHi, addrLo );
   //NANOS_INSTRUMENT( InstrumentState inst(NANOS_amWaitRequestPut); );

   gasnet_node_t src_node;
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   VERBOSE_AM( std::cerr << gasnet_mynode() << ": " << __FUNCTION__ << " from " << src_node << " addr " << (void*)addr << std::endl; );

   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( addr ) ; )
      NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_WAIT_REQ_PUT, id, sizeKey, xferSize, src_node ); )
   }

   getInstance()->_net->notifyWaitRequestPut( addr, wdId, seqNumber );
   //NANOS_INSTRUMENT( inst.close(); );
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amFreeTmpBuffer( gasnet_token_t token,
      gasnet_handlerarg_t addrLo,
      gasnet_handlerarg_t addrHi,
      gasnet_handlerarg_t wdLo,
      gasnet_handlerarg_t wdHi,
      gasnet_handlerarg_t functorLo,
      gasnet_handlerarg_t functorHi ) 
{
   void *addr = ( void * ) MERGE_ARG( addrHi, addrLo );
   WD *wd = ( WD * ) MERGE_ARG( wdHi, wdLo );
   (void) wd;
   Functor *f = ( Functor * ) MERGE_ARG( functorHi, functorLo );
   if ( f ) {
      (*f)(); 
   }
   gasnet_node_t src_node;
   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }
   //std::cerr << __FUNCTION__ << " addr " << addr << " from "<< src_node << " allocator "<< (void *) _pinnedAllocators[ src_node ] << std::endl;
   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( addr ) ; )
      NANOS_INSTRUMENT ( instr->raiseClosePtPEvent( NANOS_XFER_FREE_TMP_BUFF, id, sizeKey, xferSize, src_node ); )
   }
   getInstance()->_pinnedAllocatorsLocks[ src_node ]->acquire();
   getInstance()->_pinnedAllocators[ src_node ]->free( addr );
   getInstance()->_pinnedAllocatorsLocks[ src_node ]->release();
   //fprintf(stderr, "I CALL NOTIFY SCHED for wd %d\n", wd->getId() );
   // TODO i think this segfaults even when notify func has not been set
   // wd->notifyCopy();
   
   // XXX call notify copy wd->
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amGetStrided1D( gasnet_token_t token, void *buff, std::size_t nbytes )
{
   gasnet_node_t src_node;
   SendDataGetRequestPayload *msg = ( SendDataGetRequestPayload * ) buff;
   nanos_region_dimension_internal_t *dims = (nanos_region_dimension_internal_t *)( ((char *)buff)+ sizeof( SendDataGetRequestPayload ) );

   VERBOSE_AM( std::cerr << __FUNCTION__ << std::endl; );
   if ( gasnet_AMGetMsgSource( token, &src_node ) != GASNET_OK )
   {
      fprintf( stderr, "gasnet: Error obtaining node information.\n" );
   }

   getInstance()->_txBytes += msg->_len * msg->_count;

   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) msg->_waitObj; )
      NANOS_INSTRUMENT ( instr->raiseClosePtPEvent ( NANOS_XFER_REQ, id, sizeKey, xferSize, src_node ); )
   }

   SendDataGetRequest *req = NEW SendDataGetRequest( getInstance(), msg->_seqNumber, msg->_destAddr, msg->_origAddr, msg->_len, msg->_count, msg->_ld, msg->_waitObj, msg->_cd, dims );
   getInstance()->_net->notifyRegionMetaData( &( req->_cd ) );
   getInstance()->_net->notifyGet( req );
   VERBOSE_AM( std::cerr << __FUNCTION__ << " done." << std::endl; );
}

void GASNetAPI::amRegionMetadata(gasnet_token_t token, void *arg, std::size_t argSize ) {
   CopyData *cd = (CopyData *) arg;
   cd->setDimensions( (nanos_region_dimension_internal_t *) ( ((char*)arg) + sizeof(CopyData) ) );
   getInstance()->_net->notifyRegionMetaData( cd );
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
      { 223, (void (*)()) amGetReplyStrided1D },
      { 224, (void (*)()) amRegionMetadata }
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
    _plugin.addPinnedSegments( nodes, pinnedSegmentAddr, pinnedSegmentLen );

    uintptr_t offset = pinnedSegmentLen[ gasnet_mynode() ] / 2;
    _packSegment = NEW SimpleAllocator( ( ( uintptr_t ) pinnedSegmentAddr[ gasnet_mynode() ] ) + offset , pinnedSegmentLen[ gasnet_mynode() ] / 2 );

   if ( _net->getNodeNum() == 0)
   {
      _pinnedAllocators.reserve( nodes );
      _pinnedAllocatorsLocks.reserve( nodes );
      _seqN = NEW Atomic<unsigned int>[nodes];
      
      _net->mallocSlaves( &segmentAddr[ 1 ], _plugin.getNodeMem() );
      segmentAddr[ 0 ] = NULL;

      for ( idx = 0; idx < nodes; idx += 1)
      {
         segmentLen[ idx ] = ( idx == 0 ) ? 0 : _plugin.getNodeMem(); 
         _pinnedAllocators[idx] = NEW SimpleAllocator( ( uintptr_t ) pinnedSegmentAddr[ idx ], pinnedSegmentLen[ idx ] / 2 );
         _pinnedAllocatorsLocks[idx] =  NEW Lock( );
         new (&_seqN[idx]) Atomic<unsigned int >( 0 );
      }
      _thisNodeSegment = _pinnedAllocators[0];
      this->addSegments( nodes, segmentAddr, segmentLen );
   }
#else
   #error unimplemented
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
        // message0( "Node " << _net->getNodeNum() << " stats: Rx: " << _rxBytes << " Tx: " << _txBytes );
        // verbose0( "Node "<< _net->getNodeNum() << " closing the network." );
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
   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( remoteWdAddr ) ; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_AM_WORK, id, 0, 0, dest ); )
   }

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
   std::pair<void *, unsigned int> *rwd = NEW std::pair<void *, unsigned int> ( remoteWdAddr, (unsigned int ) peId);
   _workDoneReqs.add( rwd );
}

void GASNetAPI::_sendWorkDoneMsg ( unsigned int dest, void *remoteWdAddr, int peId )
{
   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( remoteWdAddr ) ; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_AM_WORK_DONE, id, 0, 0, dest ); )
   }
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

void GASNetAPI::_put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, std::size_t size, void *remoteTmpBuffer, unsigned int wdId, WD const &wd, Functor *f, void *hostObject, reg_t hostRegId )
{
   std::size_t sent = 0, thisReqSize;
   _txBytes += size;
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
         NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_DATA, id, sizeKey, xferSize, remoteNode ); )

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
      while ( sent < size )
      {
         thisReqSize = ( ( size - sent ) <= gasnet_AMMaxLongRequest() ) ? size - sent : gasnet_AMMaxLongRequest();

         NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
         NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
         NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )

         if ( remoteTmpBuffer != NULL )
         { 
            if ( _emitPtPEvents ) {
               NANOS_INSTRUMENT ( nanos_event_value_t xferSize = thisReqSize; )
               NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( ((uint64_t)remoteTmpBuffer) + sent ) ; )
               NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_DATA, id, sizeKey, xferSize, remoteNode ); )
            }
            //fprintf(stderr, "try to send [%d:%p=>%d:%p,%ld < %f >].\n", gasnet_mynode(), (void*)localAddr, remoteNode, (void*)remoteAddr, size, *((double *)localAddr));
            if ( gasnet_AMRequestLong16( remoteNode, 210,
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
                     ( ( sent + thisReqSize ) == size ),
                     ARG_LO( f ),
                     ARG_HI( f ),
                     ARG_LO( hostObject ),
                     ARG_HI( hostObject ),
                     hostRegId
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

void GASNetAPI::_putStrided1D ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, void *localPack, std::size_t size, std::size_t count, std::size_t ld, void *remoteTmpBuffer, unsigned int wdId, WD const &wd, Functor *f, void *hostObject, reg_t hostRegId )
{
   std::size_t sent = 0, thisReqSize;
   std::size_t realSize = size * count;
   _txBytes += realSize;
   _totalBytes += realSize;
   {
      while ( sent < realSize )
      {
         thisReqSize = ( ( realSize - sent ) <= gasnet_AMMaxLongRequest() ) ? realSize - sent : gasnet_AMMaxLongRequest();

         NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
         NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
         NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )

         if ( remoteTmpBuffer != NULL )
         { 
            if ( _emitPtPEvents ) {
               NANOS_INSTRUMENT ( nanos_event_value_t xferSize = thisReqSize; )
               NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( ((uint64_t)remoteTmpBuffer) + sent ) ; )
               NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_DATA, id, sizeKey, xferSize, remoteNode ); )
            }
            if ( gasnet_AMRequestLong16( remoteNode, 220,
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
                     ( ( sent + thisReqSize ) == realSize ),
                     ARG_LO( f ),
                     ARG_HI( f ),
                     ARG_LO( hostObject ),
                     ARG_HI( hostObject ),
                     hostRegId
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

void GASNetAPI::putStrided1D ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, void *localPack, std::size_t size, std::size_t count, std::size_t ld, unsigned int wdId, WD const &wd, void *hostObject, reg_t hostRegId ) {
   if ( gasnet_mynode() != 0 ) fatal0("Error, cant use ::put from node != than 0"); 
   void *tmp = NULL;
   while( tmp == NULL ) {
      _pinnedAllocatorsLocks[ remoteNode ]->acquire();
      tmp = _pinnedAllocators[ remoteNode ]->allocate( size * count );
      _pinnedAllocatorsLocks[ remoteNode ]->release();
      if ( tmp == NULL ) _net->poll(0);
   }
   if ( tmp == NULL ) std::cerr << "what... "<< tmp << std::endl; 
   _putStrided1D( remoteNode, remoteAddr, localAddr, localPack, size, count, ld, tmp, wdId, wd, NULL, hostObject, hostRegId );
}

void GASNetAPI::put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, std::size_t size, unsigned int wdId, WD const &wd, void *hostObject, reg_t hostRegId )
{
   if ( gasnet_mynode() != 0 ) fatal0("Error, cant use ::put from node != than 0"); 
   void *tmp = NULL;
   while( tmp == NULL ) {
      _pinnedAllocatorsLocks[ remoteNode ]->acquire();
      tmp = _pinnedAllocators[ remoteNode ]->allocate( size );
      _pinnedAllocatorsLocks[ remoteNode ]->release();
      if ( tmp == NULL ) _net->poll(0);
   }
   _put( remoteNode, remoteAddr, localAddr, size, tmp, wdId, wd, NULL, hostObject, hostRegId );
}

//Lock getLock;
#ifndef GASNET_SEGMENT_EVERYTHING
Lock getLockGlobal;
#endif

void GASNetAPI::get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, std::size_t size, volatile int *requestComplete, CopyData const &cd )
{
   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) requestComplete ; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent ( NANOS_XFER_REQ, id, sizeKey, xferSize, remoteNode ); )
   }

   unsigned int seq_number = sys.getNetwork()->getPutRequestSequenceNumber( remoteNode );
   std::size_t buffer_size = sizeof( SendDataGetRequestPayload ) + sizeof( nanos_region_dimension_internal_t ) * cd.getNumDimensions();
   char *buffer = (char *) alloca( buffer_size );
   new ( buffer ) SendDataGetRequestPayload( seq_number, localAddr, (void *)remoteAddr, size, 1, 0, (void *) requestComplete, cd );
   nanos_region_dimension_internal_t *dims = ( nanos_region_dimension_internal_t * ) ( buffer + sizeof( SendDataGetRequestPayload ) );
   ::memcpy( dims, cd.getDimensions(), sizeof( nanos_region_dimension_internal_t ) * cd.getNumDimensions() );

   if ( gasnet_AMRequestMedium0( remoteNode, 211, buffer, buffer_size ) != GASNET_OK )
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
   }

   _rxBytes += size;
   _totalBytes += size;
}

std::size_t GASNetAPI::getMaxGetStridedLen() const {
   return ( std::size_t ) gasnet_AMMaxLongReply();
}

void GASNetAPI::getStrided1D ( void *packedAddr, unsigned int remoteNode, uint64_t remoteTag, uint64_t remoteAddr, std::size_t size, std::size_t count, std::size_t ld, volatile int *requestComplete, CopyData const &cd )
{
   std::size_t thisReqSize = size * count;
   void *addr = packedAddr;

   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) requestComplete ; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent ( NANOS_XFER_REQ, id, sizeKey, xferSize, remoteNode ); )
   }

   unsigned int seq_number = sys.getNetwork()->getPutRequestSequenceNumber( remoteNode );

   std::size_t buffer_size = sizeof( SendDataGetRequestPayload ) + sizeof( nanos_region_dimension_internal_t ) * cd.getNumDimensions();
   char *buffer = (char *) alloca( buffer_size );
   new ( buffer ) SendDataGetRequestPayload( seq_number, addr, (void *)remoteAddr, size, count, ld, (void *) requestComplete, cd );
   nanos_region_dimension_internal_t *dims = ( nanos_region_dimension_internal_t * ) ( buffer + sizeof( SendDataGetRequestPayload ) );
   ::memcpy( dims, cd.getDimensions(), sizeof( nanos_region_dimension_internal_t ) * cd.getNumDimensions() );

   if ( gasnet_AMRequestMedium0( remoteNode, 221, buffer, buffer_size ) != GASNET_OK )
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", remoteNode);
   }

   _rxBytes += thisReqSize;
   _totalBytes += thisReqSize;
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

   if ( gasnet_AMRequestMedium0( dest, 209, ( void * ) masterHostname, ::strlen( masterHostname ) + 1 ) != GASNET_OK ) //+1 to add the last \0 character
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest );
   }
}

void GASNetAPI::sendRequestPut( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, std::size_t len, unsigned int wdId, WD const &wd, Functor *f, void *hostObject, reg_t hostRegId )
{
   _totalBytes += len;
   sendWaitForRequestPut( dataDest, dstAddr, wd.getHostId() );

   unsigned int seq_number = sys.getNetwork()->getPutRequestSequenceNumber( dest );

   void *tmpBuffer = NULL;
   while ( tmpBuffer == NULL ) {
      _pinnedAllocatorsLocks[ dataDest ]->acquire();
      tmpBuffer = _pinnedAllocators[ dataDest ]->allocate( len );
      _pinnedAllocatorsLocks[ dataDest ]->release();
      if ( tmpBuffer == NULL ) _net->poll(0);
   }

   SendDataPutRequestPayload msg( seq_number, (void *) origAddr, (void*) dstAddr, len, 1, 0, dataDest, wdId, tmpBuffer, &wd, f, hostObject, hostRegId );

   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( ((uint64_t)tmpBuffer) ) ; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_REQ, id, sizeKey, xferSize, dest ); )
   }
   if ( gasnet_AMRequestMedium0( dest, 214, (void *) &msg, sizeof( msg ) ) != GASNET_OK )
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GASNetAPI::sendRequestPutStrided1D( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, std::size_t len, std::size_t count, std::size_t ld, unsigned int wdId, WD const &wd, Functor *f, void *hostObject, reg_t hostRegId )
{
   _totalBytes += ( len * count );
   //NANOS_INSTRUMENT( InstrumentState inst0(NANOS_SEND_WAIT_FOR_REQ_PUT); );
   sendWaitForRequestPut( dataDest, dstAddr, wd.getHostId() );
   //NANOS_INSTRUMENT( inst0.close(); );
   //NANOS_INSTRUMENT( InstrumentState inst1(NANOS_GET_PINNED_ADDR); );
   unsigned int seq_number = sys.getNetwork()->getPutRequestSequenceNumber( dest );

   void *tmpBuffer = NULL;
   while ( tmpBuffer == NULL ) {
      _pinnedAllocatorsLocks[ dataDest ]->acquire();
      tmpBuffer = _pinnedAllocators[ dataDest ]->allocate( len * count );
      _pinnedAllocatorsLocks[ dataDest ]->release();
      if ( tmpBuffer == NULL ) _net->poll(0);
   }
   //NANOS_INSTRUMENT( inst1.close(); );

   //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_SEND_PUT_REQ); );

   SendDataPutRequestPayload msg( seq_number, (void *) origAddr, (void *) dstAddr, len, count, ld, dataDest, wdId, tmpBuffer, &wd, f, hostObject, hostRegId );
   

   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( ((uint64_t)tmpBuffer) ) ; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_REQ, id, sizeKey, xferSize, dest ); )
   }
   if ( gasnet_AMRequestMedium0( dest, 222, (void *) &msg, sizeof( msg ) ) != GASNET_OK )

   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
   //NANOS_INSTRUMENT( inst2.close(); );
}

void GASNetAPI::sendWaitForRequestPut( unsigned int dest, uint64_t addr, unsigned int wdId )
{
   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( addr ) ; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_WAIT_REQ_PUT, id, sizeKey, xferSize, dest ); )
   }

   unsigned int seq_number = sys.getNetwork()->getPutRequestSequenceNumber( dest );

   if ( gasnet_AMRequestShort4( dest, 218,
            ARG_LO( addr ), ARG_HI( addr ), wdId, seq_number ) != GASNET_OK )
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

void GASNetAPI::sendFreeTmpBuffer( void *addr, WD const *wd, Functor *f )
{
   if ( _emitPtPEvents ) {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = instr->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t sizeKey = ID->getEventKey("xfer-size"); )
      NANOS_INSTRUMENT ( nanos_event_value_t xferSize = 0; )
      NANOS_INSTRUMENT ( nanos_event_id_t id = (nanos_event_id_t) ( addr ) ; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_XFER_FREE_TMP_BUFF, id, sizeKey, xferSize, 0 ); )
   }
   if ( gasnet_AMRequestShort6( 0, 219,
            ARG_LO( addr ), ARG_HI( addr ), ARG_LO( wd ), ARG_HI( wd ), ARG_LO( f ), ARG_HI( f ) ) != GASNET_OK )
   {
      fprintf(stderr, "gasnet: Error sending a message to node 0.\n");
   }
}

void GASNetAPI::sendRegionMetadata( unsigned int dest, CopyData *cd ) {
   std::size_t data_size = sizeof(CopyData) + cd->getNumDimensions() * sizeof(nanos_region_dimension_internal_t);
   char *buffer = (char *) alloca(data_size);

   ::memcpy(buffer, cd, sizeof(CopyData) );
   ::memcpy(buffer + sizeof(CopyData), cd->getDimensions(), cd->getNumDimensions() * sizeof(nanos_region_dimension_internal_t));

   if ( gasnet_AMRequestMedium0( dest, 224, (void *) buffer, data_size ) != GASNET_OK )
   {
      fprintf(stderr, "gasnet: Error sending a message to node %d.\n", dest);
   }
}

std::size_t GASNetAPI::getRxBytes()
{
   return _rxBytes;
}

std::size_t GASNetAPI::getTxBytes()
{
   return _txBytes;
}

std::size_t GASNetAPI::getTotalBytes()
{
   return _totalBytes;
}



SimpleAllocator *GASNetAPI::getPackSegment() const {
   return _packSegment;
}

void *GASNetAPI::allocateReceiveMemory( std::size_t len ) {
   void *addr = NULL;
   do {
      getLockGlobal.acquire();
      addr = _thisNodeSegment->allocate( len );
      getLockGlobal.release();
      if ( addr == NULL ) myThread->idle();
   } while (addr == NULL);
   return addr;
}

void GASNetAPI::freeReceiveMemory( void * addr ) {
   getLockGlobal.acquire();
   _thisNodeSegment->free( addr );
   getLockGlobal.release();
}

GASNetAPI::FreeBufferRequest::FreeBufferRequest(void *addr, WD const *w, Functor *f ) : address( addr ), wd ( w ), functor( f ) {
}

void GASNetAPI::addSegments( unsigned int numSegments, void **segmentAddr, std::size_t *segmentSize ) {
   unsigned int idx;
   _numSegments = numSegments;
   _segmentAddrList = NEW void *[ numSegments ];
   _segmentLenList = NEW std::size_t[ numSegments ];

   for ( idx = 0; idx < numSegments; idx += 1)
   {
      _segmentAddrList[ idx ] = segmentAddr[ idx ];
      _segmentLenList[ idx ] = segmentSize[ idx ];
   }
}

void * GASNetAPI::getSegmentAddr( unsigned int idx ) {
   return _segmentAddrList[ idx ];
}

std::size_t GASNetAPI::getSegmentLen( unsigned int idx ) {
   return _segmentLenList[ idx ];
}

unsigned int GASNetAPI::getNumNodes() const {
   return gasnet_nodes();
}

unsigned int GASNetAPI::getNodeNum() const {
   return gasnet_mynode();
}
