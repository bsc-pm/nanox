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
#include "schedule.hpp"
#include "system.hpp"

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
         NANOS_INSTRUMENT ( instr->raiseOpenPtPEventNkvs( NANOS_WD_REMOTE, id, 0, NULL, NULL, dest ); )

         _api->sendWorkMsg( dest, work, dataSize, wdId, numPe, argSize, arg, xlate, arch, remoteWd );
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
      _api->put( remoteNode, remoteAddr, localAddr, size, wdId, wd );
   }
}

void Network::putStrided1D ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, void *localPack, std::size_t size, std::size_t count, std::size_t ld, unsigned int wdId, WD const &wd )
{
   if ( _api != NULL )
   {
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


void Network::sendRequestPut( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, std::size_t len, unsigned int wdId, WD const &wd )
{
   if ( _api != NULL )
   {
      _api->sendRequestPut( dest, origAddr, dataDest, dstAddr, len, wdId, wd );
   }
}

void Network::sendRequestPutStrided1D( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, std::size_t len, std::size_t count, std::size_t ld, unsigned int wdId, WD const &wd )
{
   if ( _api != NULL )
   {
      _api->sendRequestPutStrided1D( dest, origAddr, dataDest, dstAddr, len, count, ld, wdId, wd );
   }
}

void Network::setNewMasterDirectory(NewRegionDirectory *dir)
{
   if ( _api != NULL) 
   {
      _api->setNewMasterDirectory( dir );
   }
}

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
