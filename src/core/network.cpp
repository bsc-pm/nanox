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


#include <iostream>
#include <cstring>
#include "network_decl.hpp"
#include "requestqueue.hpp"
#include "schedule.hpp"
#include "system.hpp"
#include "deviceops.hpp"

#define VERBOSE_COMPLETION 0

using namespace nanos;

Lock Network::_nodeLock;
Atomic<uint64_t> Network::_nodeRCAaddr;
Atomic<uint64_t> Network::_nodeRCAaddrOther;

Network::Network () : _numNodes( 1 ), _api( (NetworkAPI *) 0 ), _nodeNum( 0 ), _masterHostname ( NULL ) {}
Network::~Network () {}

void Network::setAPI ( NetworkAPI *api )
{
   _api = api;
}

NetworkAPI *Network::getAPI ()
{
   return _api;
}

void Network::setNumNodes ( unsigned int numNodes )
{
   _numNodes = numNodes;
}

unsigned int Network::getNumNodes () const
{
   return _numNodes;
}

void Network::setNodeNum ( unsigned int nodeNum )
{
   _nodeNum = nodeNum;
}

unsigned int Network::getNodeNum () const
{
   return _nodeNum;
}

void Network::initialize()
{
   //   ensure ( _api != NULL, "No network api loaded." );
   if ( _api != NULL )
      _api->initialize( this );

   _putRequestSequence = NEW Atomic<unsigned int>[ getNumNodes() ];
   for ( unsigned int i = 0; i < getNumNodes(); i += 1 ) {
      new ( &_putRequestSequence[ i ] ) Atomic<unsigned int>( 0 );
   }
}

void Network::finalize()
{
   if ( _api != NULL )
   {
      _api->finalize();
   }
}

void Network::poll( unsigned int id)
{
   //   ensure ( _api != NULL, "No network api loaded." );
   checkDeferredWorkReqs();
   processRequestsDelayedBySeqNumber();
   SendDataRequest * req = _dataSendRequests.tryFetch();
   if ( req ) {
      _api->processSendDataRequest( req );
   }
   if (_api != NULL /*&& (id >= _pollingMinThd && id <= _pollingMaxThd) */)
      _api->poll();
}

void Network::sendExitMsg( unsigned int nodeNum )
{
   //  ensure ( _api != NULL, "No network api loaded." );
   if ( _nodeNum == MASTER_NODE_NUM )
   {
      _api->sendExitMsg( nodeNum );
   }
}

void Network::sendWorkMsg( unsigned int dest, void ( *work ) ( void * ), unsigned int dataSize, unsigned int wdId, unsigned int numPe, std::size_t argSize, char * arg, void ( *xlate ) ( void *, void * ), int arch, void *remoteWd )
{
   //  ensure ( _api != NULL, "No network api loaded." );
   if ( _api != NULL )
   {
      if (work == NULL)
      {
         std::cerr << "ERROR: no work to send (work=NULL)" << std::endl;
      }
      if ( _nodeNum == MASTER_NODE_NUM )
      {
         NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
         NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) wdId) ) ; )
         NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent( NANOS_WD_REMOTE, id, 0, 0, dest ); )
      
         std::size_t expectedData = _sentWdData.getSentData( wdId );
         _api->sendWorkMsg( dest, work, dataSize, wdId, numPe, argSize, arg, xlate, arch, remoteWd, expectedData );
      }
      else
      {
         std::cerr << "tried to send work from a node != 0" << std::endl;
      }
   }
}

void Network::sendWorkDoneMsg( unsigned int nodeNum, void *remoteWdAddr, int peId )
{
   //  ensure ( _api != NULL, "No network api loaded." );
   if ( _api != NULL )
   {
      //NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      //NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) remoteWdAddr)  ) ; )
      //NANOS_INSTRUMENT ( instr->raiseOpenPtPEventNkvs( NANOS_WD_REMOTE, id, 0, NULL, NULL, 0 ); )
      if ( _nodeNum != MASTER_NODE_NUM )
      {
         _api->sendWorkDoneMsg( nodeNum, remoteWdAddr, peId );
      }
   }
}

void Network::notifyWorkDone ( unsigned int nodeNum, void *remoteWdAddr, int peId)
{
   //NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   //NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) remoteWdAddr) ) ; )
   //NANOS_INSTRUMENT ( instr->raiseClosePtPEventNkvs( NANOS_WD_REMOTE, id, 0, NULL, NULL, nodeNum ); )

   ( (WD *) remoteWdAddr )->notifyOutlinedCompletion();
}

void Network::put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, std::size_t size, unsigned int wdId, WD const &wd )
{
   if ( _api != NULL )
   {
      _sentWdData.addSentData( wdId, size );
      _api->put( remoteNode, remoteAddr, localAddr, size, wdId, wd );
   }
}

void Network::putStrided1D ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, void *localPack, std::size_t size, std::size_t count, std::size_t ld, unsigned int wdId, WD const &wd )
{
   if ( _api != NULL )
   {
      _sentWdData.addSentData( wdId, size * count );
      _api->putStrided1D( remoteNode, remoteAddr, localAddr, localPack, size, count, ld, wdId, wd );
   }
}

void Network::get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, std::size_t size, volatile int *req )
{
   if ( _api != NULL )
   {
      _api->get( localAddr, remoteNode, remoteAddr, size, req );
   }
}

void Network::getStrided1D ( void *packedAddr, unsigned int remoteNode, uint64_t remoteTag, uint64_t remoteAddr, std::size_t size, std::size_t count, std::size_t ld, volatile int* req )
{
   if ( _api != NULL )
   {
      _api->getStrided1D( packedAddr, remoteNode, remoteTag, remoteAddr, size, count, ld, req );
   }
}

void * Network::malloc ( unsigned int remoteNode, std::size_t size )
{
   mallocWaitObj request;

   request.complete = 0;
   request.resultAddr = NULL;

   if ( _api != NULL )
   {
      _api->malloc( remoteNode, size, ( void * ) &request );

      while ( ( (volatile int) request.complete ) == 0 )
      {
         poll( /*myThread->getId()*/0 );
      }
   }

   return request.resultAddr;
}

void Network::mallocSlaves ( void **addresses, std::size_t size )
{
   unsigned int index;
   mallocWaitObj request[ _numNodes - 1 ];
   if ( _api != NULL )
   {

      //std::cerr << "malloc on slaves..." << std::endl;
      for ( index = 0; index < ( _numNodes - 1 ); index += 1) {
         request[ index ].complete = 0;
         request[ index ].resultAddr = NULL;
         _api->malloc( index+1, size, ( void * ) &( request[ index ] ) );
      }
      //std::cerr << "malloc on slaves... wait responses" << std::endl;

      for ( index = 0; index < ( _numNodes - 1 ); index += 1) {
         while ( ( (volatile int) request[ index ].complete ) == 0 )
         {
            poll( /*myThread->getId()*/0 );
         }
	 addresses[ index ] = request[ index ].resultAddr;
      }
      //std::cerr << "malloc on slaves... complete" << std::endl;
   }
}

void Network::memFree ( unsigned int remoteNode, void *addr )
{
   if ( _api != NULL )
   {
      _api->memFree( remoteNode, addr );
   }
}

void Network::memRealloc ( unsigned int remoteNode, void *oldAddr, std::size_t oldSize, void *newAddr, std::size_t newSize )
{
   if ( _api != NULL )
   {
      _api->memRealloc( remoteNode, oldAddr, oldSize, newAddr, newSize );
   }
}

void Network::notifyMalloc( unsigned int remoteNode, void * addr, mallocWaitObj *request )
{
   //std::cerr << "recv malloc response from "<< remoteNode << std::endl;
   request->resultAddr = addr;
   request->complete = 1;
}

void Network::nodeBarrier()
{
   if ( _api != NULL )
   {
      _api->nodeBarrier();
   }
}

void Network::setMasterHostname( char *name )
{
   //  _masterHostname = std::string( name );
   if ( _masterHostname == NULL )
      _masterHostname = new char[ ::strlen( name ) + 1 ];
   ::bzero( _masterHostname, ::strlen( name ) + 1 );
   ::memcpy(_masterHostname, name, ::strlen( name ) );
   _masterHostname[ ::strlen( name ) ] = '\0';
}

//const std::string & Network::getMasterHostname() const
const char * Network::getMasterHostname() const
{
   return _masterHostname; 
}


void Network::sendRequestPut( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, std::size_t len, unsigned int wdId, WD const &wd, Functor *f )
{
   if ( _api != NULL )
   {
      _sentWdData.addSentData( wdId, len );
      _api->sendRequestPut( dest, origAddr, dataDest, dstAddr, len, wdId, wd, f );
   }
}

void Network::sendRequestPutStrided1D( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, std::size_t len, std::size_t count, std::size_t ld, unsigned int wdId, WD const &wd, Functor *f )
{
   if ( _api != NULL )
   {
      _sentWdData.addSentData( wdId, len * count );
      _api->sendRequestPutStrided1D( dest, origAddr, dataDest, dstAddr, len, count, ld, wdId, wd, f );
   }
}

//void Network::setNewMasterDirectory(NewRegionDirectory *dir)
//{
//   if ( _api != NULL) 
//   {
//      _api->setNewMasterDirectory( dir );
//   }
//}

std::size_t Network::getTotalBytes()
{
   std::size_t result = 0;
   if ( _api != NULL )
   {
      result = _api->getTotalBytes();
   }
   return result;
}

void Network::enableCheckingForDataInOtherAddressSpaces()
{
   _checkForDataInOtherAddressSpaces = true;
}

bool Network::doIHaveToCheckForDataInOtherAddressSpaces() const
{
   return _checkForDataInOtherAddressSpaces;
}

SimpleAllocator *Network::getPackerAllocator() const
{
   SimpleAllocator *res = NULL;
   if ( _api != NULL)
   {
      res = _api->getPackSegment();
   }
   //std::cerr <<" PACK SEGMENT IS " << res << std::endl;
   return res;
}
std::size_t Network::getMaxGetStridedLen() const {
   std::size_t result = 0;
   if ( _api != NULL ) {
      result = _api->getMaxGetStridedLen();
   }
   return result;
}

void *Network::allocateReceiveMemory( std::size_t len ) {
   void *addr = NULL;
   if ( _api != NULL ) {
      addr = _api->allocateReceiveMemory( len );
   }
   return addr;
}

void Network::freeReceiveMemory( void * addr ) {
   if ( _api != NULL ) {
      _api->freeReceiveMemory( addr );
   }
}


// API -> system mechanisms
Network::ReceivedWDData::ReceivedWDData() : _recvWdDataMap(), _lock(), _receivedWDs( 0 ) {
}

Network::ReceivedWDData::~ReceivedWDData() {
}

void Network::ReceivedWDData::addData( unsigned int wdId, std::size_t size ) {
   _lock.acquire();
   struct recvDataInfo &info = _recvWdDataMap[ wdId ];
   info._count += size;
   if ( info._wd != NULL && info._expected == info._count ) {
      WD *wd = info._wd;
      if ( _recvWdDataMap.erase( wdId ) != 1) std::cerr <<"Error removing from map: "<<__FUNCTION__<< " @ " << __FILE__<<":"<<__LINE__<< std::endl;
      _lock.release();
      //release wd
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) wdId)  )  ; )
      NANOS_INSTRUMENT ( instr->createDeferredPtPEnd ( *wd, NANOS_WD_REMOTE, id, 0, 0, 0 ); )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent ( NANOS_WD_DOMAIN, (nanos_event_id_t) wdId, 0, 0 );)
      sys.submit( *wd );
      _receivedWDs++;
      //std::cerr <<"["<< gasnet_mynode()<< "] release wd (by data) new seq is " << _recvSeqN.value()   << std::endl;
   } else {
      _lock.release();
   }
}

void Network::ReceivedWDData::addWD( unsigned int wdId, WorkDescriptor *wd, std::size_t expectedData ) {
   _lock.acquire();
   struct recvDataInfo &info = _recvWdDataMap[ wdId ];
   info._wd = wd;
   info._expected = expectedData;
   //std::cerr <<"["<< gasnet_mynode()<< "] addWD with expected data: " << expectedData << " current count: " << info._count  << std::endl;
   if ( info._expected == info._count ) {
      if ( _recvWdDataMap.erase( wdId ) != 1) std::cerr <<"Error removing from map: "<<__FUNCTION__<< " @ " << __FILE__<<":"<<__LINE__<< std::endl;
      _lock.release();
      //release wd
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) wdId)  )  ; )
      NANOS_INSTRUMENT ( instr->createDeferredPtPEnd ( *wd, NANOS_WD_REMOTE, id, 0, 0, 0 ); )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent ( NANOS_WD_DOMAIN, (nanos_event_id_t) wdId, 0, 0 );)
      sys.submit( *wd );
      _receivedWDs++;
   //std::cerr <<"["<< gasnet_mynode()<< "] release wd (by wd) new seq is " << _recvSeqN.value()   << std::endl;
   } else {
      _lock.release();
   }
}

unsigned int Network::ReceivedWDData::getReceivedWDsCount() const {
   return _receivedWDs.value();
}

Network::SentWDData::SentWDData() : _sentWdDataMap(), _lock() {
}

Network::SentWDData::~SentWDData() {
}

void Network::SentWDData::addSentData( unsigned int wdId, std::size_t sentData ) {
   _lock.acquire();
   _sentWdDataMap[ wdId ] += sentData;//assumes that if no data was yet sent, the elem is initialized to 0
   _lock.release();
}

std::size_t Network::SentWDData::getSentData( unsigned int wdId ) {
   _lock.acquire();
   std::size_t wdSentData = _sentWdDataMap[ wdId ];
   if ( _sentWdDataMap.erase( wdId ) != 1) std::cerr <<"Error removing from map: "<<__FUNCTION__<< " @ " << __FILE__<<":"<<__LINE__<< std::endl;
   _lock.release();
   return wdSentData;
}

void Network::notifyWork(std::size_t expectedData, WD *delayedWD, unsigned int delayedSeq) {
   if ( _recvWdData.getReceivedWDsCount() == delayedSeq )
   {
      _recvWdData.addWD( delayedWD->getId(), delayedWD, expectedData );
      checkDeferredWorkReqs();
   } else { //not expected seq number, enqueue
      _deferredWorkReqsLock.acquire();
      _deferredWorkReqs.push_back( std::pair< unsigned int, std::pair < WD*, std::size_t > >( delayedSeq, std::pair< WD*, std::size_t > ( delayedWD, expectedData ) ) );
      _deferredWorkReqsLock.release();
   }
}

void Network::checkDeferredWorkReqs()
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
         if (dwd.first == _recvWdData.getReceivedWDsCount() ) 
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

void Network::notifyPut( unsigned int from, unsigned int wdId, std::size_t len, std::size_t count, std::size_t ld, uint64_t realTag ) {
   // TODO if ( doIHaveToCheckForDataInOtherAddressSpaces() ) {
   // TODO    invalidateDataFromDevice( (uint64_t) realTag, len, count, ld );
   // TODO }
   //std::cerr << "ADD wd data for wd "<< wdId << " len " << len*count << std::endl;
   _recvWdData.addData( wdId, len*count );
   if ( from != 0 ) { /* check for delayed putReqs or gets */
      _waitingPutRequestsLock.acquire();
      std::set<void *>::iterator it;
      if ( ( it = _waitingPutRequests.find( (void*)realTag ) ) != _waitingPutRequests.end() )
      {
         void *destAddr = *it;
         _waitingPutRequests.erase( it );
         _delayedPutReqsLock.acquire();
         if ( !_delayedPutReqs.empty() ) {
            for ( std::list<SendDataRequest *>::iterator putReqsIt = _delayedPutReqs.begin(); putReqsIt != _delayedPutReqs.end(); ) {
               if ( (*putReqsIt)->getOrigAddr() == destAddr ) {
                  _dataSendRequests.add( *putReqsIt );
                  putReqsIt = _delayedPutReqs.erase( putReqsIt );
               } else {
                  putReqsIt++;
               }
            }
         }
         _delayedPutReqsLock.release();
      }
      else
      {
         _receivedUnmatchedPutRequests.insert( (void *) realTag );
      }
      _waitingPutRequestsLock.release();
   }
}

void Network::notifyRequestPut( SendDataRequest *req ) {
   if ( checkPutRequestSequenceNumber( 0 ) == req->getSeqNumber() ) {
      processSendDataRequest( req );
   } else {
      _delayedBySeqNumberPutReqsLock.acquire();
      _delayedBySeqNumberPutReqs.push_back( UnorderedRequest( req ) );
      _delayedBySeqNumberPutReqsLock.release();
   }
}

void Network::notifyGet( SendDataRequest *req ) {
   if ( checkPutRequestSequenceNumber( 0 ) ==  req->getSeqNumber() ) {
      processSendDataRequest( req );
   } else {
      _delayedBySeqNumberPutReqsLock.acquire();
      _delayedBySeqNumberPutReqs.push_back( UnorderedRequest( req ) );
      _delayedBySeqNumberPutReqsLock.release();
   }
}

void Network::notifyWaitRequestPut( void *addr, unsigned int wdId, unsigned int seqNumber ) {
   if ( checkPutRequestSequenceNumber( 0 ) == seqNumber ) {
      processWaitRequestPut( addr, seqNumber );
   } else {
      _delayedBySeqNumberPutReqsLock.acquire();
      _delayedBySeqNumberPutReqs.push_back( UnorderedRequest( addr, seqNumber ) );
      _delayedBySeqNumberPutReqsLock.release();
   }
}

void Network::processWaitRequestPut( void *addr, unsigned int seqNumber ) {
   _waitingPutRequestsLock.acquire();
   std::set< void * >::iterator it;
   if ( _receivedUnmatchedPutRequests.empty() || ( it = _receivedUnmatchedPutRequests.find( addr ) ) == _receivedUnmatchedPutRequests.end() ) {
      _waitingPutRequests.insert( addr );
   } else {
      _receivedUnmatchedPutRequests.erase( it );
   }
   _waitingPutRequestsLock.release();
   updatePutRequestSequenceNumber( 0, seqNumber );
}

void Network::processSendDataRequest( SendDataRequest *req ) {
   _waitingPutRequestsLock.acquire();
   if ( _waitingPutRequests.find( req->getOrigAddr() ) != _waitingPutRequests.end() ) //we have to wait 
   {
      _waitingPutRequestsLock.release();
      _delayedPutReqsLock.acquire();
      _delayedPutReqs.push_back( req );
      _delayedPutReqsLock.release();
   } else {
      _waitingPutRequestsLock.release();
      _api->processSendDataRequest( req );
   }
   updatePutRequestSequenceNumber( 0, req->getSeqNumber() );
}


void Network::processRequestsDelayedBySeqNumber() {
   if ( _delayedBySeqNumberPutReqsLock.tryAcquire() ) {
      for ( std::list< UnorderedRequest >::iterator it = _delayedBySeqNumberPutReqs.begin(); it != _delayedBySeqNumberPutReqs.end(); ) {
         if ( it->_addr != ( void * ) 0 ) {
            //its a waitRequest
            if ( checkPutRequestSequenceNumber( 0 ) == it->_seqNumber ) {
               processWaitRequestPut( it->_addr, it->_seqNumber );
               _delayedBySeqNumberPutReqs.erase( it );
               it = _delayedBySeqNumberPutReqs.begin();
            } else {
               it++;
            }
         } else {
            //its a RequestPut/Get
            if ( checkPutRequestSequenceNumber( 0 ) == it->_req->getSeqNumber() ) {
               processSendDataRequest( it->_req );
               _delayedBySeqNumberPutReqs.erase( it );
               it = _delayedBySeqNumberPutReqs.begin();
            } else {
               it++;
            }
         }
      }
      _delayedBySeqNumberPutReqsLock.release();
   }
}

SendDataRequest::SendDataRequest( NetworkAPI *api, unsigned int seqNumber, void *origAddr, void *destAddr, std::size_t len, std::size_t count, std::size_t ld, unsigned int dst, unsigned int wdId ) :
   _api( api ), _seqNumber( seqNumber ), _origAddr( origAddr ), _destAddr( destAddr ), _len( len ), _count( count ), _ld( ld ), _destination( dst ), _wdId( wdId ) {
}

SendDataRequest::~SendDataRequest() {
}

void SendDataRequest::doSend() {
   // TODO if ( doIHaveToCheckForDataInOtherAddressSpaces() ) {
   // TODO    getDataFromDevice( (uint64_t) _origAddr, _len, _count, _ld );
   // TODO }
   if ( _ld == 0 ) {
      //NANOS_INSTRUMENT( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("cache-copy-in") );
      //NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenStateAndBurst( NANOS_MEM_TRANSFER_IN, key, (nanos_event_value_t) _len) );
      doSingleChunk();
      //NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseStateAndBurst( key ) );
   } else {
      char *localPack, *origAddrPtr = (char*) _origAddr;

      //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_STRIDED_COPY_PACK); );
      _api->getPackSegment()->lock();
      localPack = ( char * ) _api->getPackSegment()->allocate( _len * _count );
      if ( localPack == NULL ) { fprintf(stderr, "ERROR!!! could not get an addr to pack strided data\n" ); }
      _api->getPackSegment()->unlock();

      for ( unsigned int i = 0; i < _count; i += 1 ) {
         memcpy( &localPack[ i * _len ], &origAddrPtr[ i * _ld ], _len );
      }
      //NANOS_INSTRUMENT( inst2.close(); );

      doStrided( localPack );

      _api->getPackSegment()->lock();
      _api->getPackSegment()->free( localPack );
      _api->getPackSegment()->unlock();
   }
}
void *SendDataRequest::getOrigAddr() const {
   return _origAddr;
}

unsigned int SendDataRequest::getDestination() const {
   return _destination;
}

unsigned int SendDataRequest::getWdId() const {
   return _wdId;
}

unsigned int SendDataRequest::getSeqNumber() const {
   return _seqNumber;
}

void Network::invalidateDataFromDevice( uint64_t addr, std::size_t len, std::size_t count, std::size_t ld ) {
   //NewDirectory::LocationInfoList locations;
   //unsigned int currentVersion;
   //nanos_region_dimension_internal_t aDimension = { len, 0, len };
   //CopyData cd( addr, NANOS_SHARED, true, true, 1, &aDimension, 0);
   //Region reg = NewRegionDirectory::build_region( cd );
   //getInstance()->_newMasterDir->lock();
   ////getInstance()->_newMasterDir->masterRegisterAccess( reg, true, true /* will increase version number */, 0, addr /*this is currently unused */, locations );
   //getInstance()->_newMasterDir->masterGetLocation( reg, locations, currentVersion ); 
   //getInstance()->_newMasterDir->addAccess( reg, 0, currentVersion + 1 ); 
   //getInstance()->_newMasterDir->unlock();
}

void Network::getDataFromDevice( uint64_t addr, std::size_t len, std::size_t count, std::size_t ld ) {
   //NewDirectory::LocationInfoList locations;
   //unsigned int currentVersion;
   //nanos_region_dimension_internal_t aDimension = { len, 0, len };
   //CopyData cd( addr, NANOS_SHARED, true, true, 1, &aDimension, 0);
   //Region reg = NewRegionDirectory::build_region( cd );
   //getInstance()->_newMasterDir->lock();
   //getInstance()->_newMasterDir->masterGetLocation( reg, locations, currentVersion ); 
   //getInstance()->_newMasterDir->addAccess( reg, 0, currentVersion ); 
   //getInstance()->_newMasterDir->unlock();
   //for ( NewDirectory::LocationInfoList::iterator it = locations.begin(); it != locations.end(); it++ ) {
   //   if (!it->second.isLocatedIn( 0 ) ) { 
   //      unsigned int loc = it->second.getFirstLocation();
   //      //std::cerr << "Houston, we have a problem, data is not in Host and we need it back. HostAddr: " << (void *) (((it->first)).getFirstValue()) << it->second << std::endl;
   //      sys.getCaches()[ loc ]->syncRegion( it->first /*, it->second.getAddressOfLocation( loc )*/ );
   //      //std::cerr <<"[" << gasnet_mynode() << "] Sync data to host mem, got " << *((double *) it->first.getFirstValue())<< " addr is " << (void *) it->first.getFirstValue() << std::endl;
   //   }
   //  //else { /*if ( sys.getNetwork()->getNodeNum() == 0)*/ std::cerr << "["<<sys.getNetwork()->getNodeNum()<<"]  All ok, checked directory " << _newMasterDir  <<" location is " << it->second << std::endl; }
   //}
}

unsigned int Network::getPutRequestSequenceNumber( unsigned int dest ) {
   return _putRequestSequence[ dest ]++;
}

unsigned int Network::checkPutRequestSequenceNumber( unsigned int dest ) const {
   return _putRequestSequence[ dest ].value();
}

bool Network::updatePutRequestSequenceNumber( unsigned int dest, unsigned int value ) {
   return _putRequestSequence[ dest ].cswap( value, value + 1 );
}

GetRequest::GetRequest( char* hostAddr, std::size_t size, char *recvAddr, DeviceOps *ops, Functor *f ) : _complete(0),
   _hostAddr( hostAddr ), _size( size ), _recvAddr( recvAddr ), _ops( ops ), _f( f ) {
}

GetRequest::~GetRequest() {
}

void GetRequest::complete() {
   (*_f)();
   _complete = 1;
}

bool GetRequest::isCompleted() const {
   return _complete == 1;
}

void GetRequest::clear() {
   ::memcpy( _hostAddr, _recvAddr, _size );
   if ( VERBOSE_COMPLETION ) {
      std::cerr << "Completed copyOut request, hostAddr="<< (void*)_hostAddr <<" ["<< *((double*) _hostAddr) <<"] ops=" << (void *) _ops << std::endl;
   }
   sys.getNetwork()->freeReceiveMemory( _recvAddr );
   _ops->completeOp();
}

GetRequestStrided::GetRequestStrided( char* hostAddr, std::size_t size, std::size_t count, std::size_t ld, char *recvAddr, DeviceOps *ops, Functor *f, Packer *packer ) :
   GetRequest( hostAddr, size, recvAddr, ops, f ), _count( count ), _ld( ld ), _packer( packer ) {
}

GetRequestStrided::~GetRequestStrided() {
}

void GetRequestStrided::clear() {
   //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_STRIDED_COPY_UNPACK); );
   for ( unsigned int j = 0; j < _count; j += 1 ) {
      ::memcpy( &_hostAddr[ j  * _ld ], &_recvAddr[ j * _size ], _size );
   }
   if ( VERBOSE_COMPLETION ) {
      std::cerr << "Completed copyOutStrided request, hostAddr="<< (void*)_hostAddr <<" ["<< *((double*) _hostAddr) <<"] ops=" << (void *) _ops << std::endl;
   }
   //NANOS_INSTRUMENT( inst2.close(); );
   _packer->free_pack( (uint64_t) _hostAddr, _size, _count, _recvAddr );
   _ops->completeOp();
}
