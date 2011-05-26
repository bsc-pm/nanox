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
#include "network.hpp"
#include "schedule.hpp"
#include "system.hpp"
#include "directory_decl.hpp"
#include "clusterthread.hpp"

using namespace nanos;

Lock Network::_nodeLock;
Atomic<uint64_t> Network::_nodeRCAaddr;
Atomic<uint64_t> Network::_nodeRCAaddrOther;

Network::Network () : _numNodes( 1 ), _api( (NetworkAPI *) 0 ), _nodeNum( 0 ), _notify( NULL ),
                      _malloc_return( NULL ), _malloc_complete( NULL ), _masterHostname ( NULL ), _pollingMinThd(999), _pollingMaxThd( 999 ) {}
Network::~Network ()
{
   if ( _notify != NULL )
      delete _notify;
   if ( _malloc_return != NULL )
      delete _malloc_return;
   if ( _malloc_complete != NULL )
      delete _malloc_complete;
}

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
   unsigned int i;
   unsigned int totalPEs = numNodes * 2 ; //( sys.getNumPEs() + 1 );

   _numNodes = numNodes;
   
//std::cerr << "initializing network with " << sys.getNumPEs() << " PEs per mode" << std::endl;
   _notify = new volatile unsigned int[ totalPEs ];
   _malloc_return = new void *[ totalPEs ];
   _malloc_complete = new bool[ totalPEs ];

   for (i = 0; i < totalPEs; i++)
   {
      _notify[ i ] = 0;
      _malloc_complete[ i ] = false;
   }

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
      //std::cerr << "[" << _nodeNum << "] => " << nodeNum << " " << __FUNCTION__ << std::endl;
      _api->sendExitMsg( nodeNum );
   }
}

void Network::sendWorkMsg( unsigned int dest, void ( *work ) ( void * ), unsigned int dataSize, unsigned int wdId, unsigned int numPe, size_t argSize, char * arg, void ( *xlate ) ( void *, void * ), int arch, void *remoteWd )
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
       //  std::cerr << "work sent to " << dest << std::endl;

         NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
         NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) numPe) << 32 ) + dest; )
         NANOS_INSTRUMENT ( instr->raiseOpenPtPEventNkvs( NANOS_WD_REMOTE, id, 0, NULL, NULL, dest ); )

      //std::cerr << "[" << _nodeNum << "] => " << dest << " " << __FUNCTION__ << std::endl;
         _api->sendWorkMsg( dest, work, dataSize, wdId, numPe, argSize, arg, xlate, arch, remoteWd );
         _notify[ dest * 2 /*sys.getNumPEs()*/ + numPe ] = 1; //FIXME: hardcoded for 1 GPU + 1 SMP per node
      }
      else
      {
         std::cerr << "tried to send work from a node != 0" << std::endl;
      }
   }
}

bool Network::isWorking(unsigned int dest, unsigned int numPe) const
{
   return ( _notify[ dest * 2 /*sys.getNumPEs()*/ + numPe ] == 1 );  //FIXME: hardcoded for 1 GPU + 1 SMP per node
}

void Network::sendWorkDoneMsg( unsigned int nodeNum, void *remoteWdAddr, int peId )
{
 //  ensure ( _api != NULL, "No network api loaded." );
   if ( _api != NULL )
   {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) 0) << 32 ) + _nodeNum; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEventNkvs( NANOS_WD_REMOTE, id, 0, NULL, NULL, 0 ); )
      if ( _nodeNum != MASTER_NODE_NUM )
      {
      //std::cerr << "[" << _nodeNum << "] => " << nodeNum << " " << __FUNCTION__ << std::endl;
         _api->sendWorkDoneMsg( nodeNum, remoteWdAddr, peId );
      }
   }
}

void Network::notifyWorkDone ( unsigned int nodeNum, void *remoteWdAddr, int peId)
{
   //std::cerr << "completed work from " << nodeNum << " : " << numPe << std::endl;
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) 0) << 32 ) + nodeNum; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEventNkvs( NANOS_WD_REMOTE, id, 0, NULL, NULL, nodeNum ); )

   //_notify[ nodeNum * 2 /*sys.getNumPEs()*/ + 1 ] = 0;
   //_thds[ nodeNum - 1 ]->completeWDGPU(  remoteWdAddr );
   if ( peId == 0) 
   _thds[ nodeNum - 1 ]->completeWDSMP_2( remoteWdAddr );
   else if ( peId == 1)
   _thds[ nodeNum - 1 ]->completeWDGPU_2( remoteWdAddr );
   else
    std::cerr << "unhandled peid" << std::endl;
}

void Network::put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size )
{
   if ( _api != NULL )
{
      //std::cerr << "[" << _nodeNum << ":" << localAddr << "] => " << remoteNode << ":" << ((void *) remoteAddr) << " " << __FUNCTION__ << std::endl;
      _api->put( remoteNode, remoteAddr, localAddr, size );
}
}

void Network::get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, size_t size )
{
   if ( _api != NULL )
{
      //std::cerr << "[" << _nodeNum << ":" << localAddr << "] => " << remoteNode << ":" << ((void *) remoteAddr) << " " << __FUNCTION__ << std::endl;
      _api->get( localAddr, remoteNode, remoteAddr, size );
}
}

void * Network::malloc ( unsigned int remoteNode, size_t size, unsigned int id )
{
   void * result = NULL;

   if ( _api != NULL )
   {
      _api->malloc( remoteNode, size, id );

      while ( _malloc_complete[ remoteNode * 2 /*sys.getNumPEs()*/ + id ] == false )
      {
         poll( myThread->getId() );
      }

      result = _malloc_return[ remoteNode * 2/*sys.getNumPEs()*/ + id ];
      _malloc_complete[ remoteNode * 2/*sys.getNumPEs()*/ + id ] = false;
   }

   return result;
}

void Network::memFree ( unsigned int remoteNode, void *addr )
{
   if ( _api != NULL )
   {
      _api->memFree( remoteNode, addr );
   }
}

void Network::memRealloc ( unsigned int remoteNode, void *oldAddr, size_t oldSize, void *newAddr, size_t newSize )
{
   if ( _api != NULL )
   {
      _api->memRealloc( remoteNode, oldAddr, oldSize, newAddr, newSize );
   }
}

void Network::notifyMalloc( unsigned int remoteNode, void * addr, unsigned int id )
{
   _malloc_return[ remoteNode * 2/* sys.getNumPEs()*/ + id ] = addr;
   _malloc_complete[ remoteNode * 2 /*sys.getNumPEs()*/ + id ] = true;
}

void Network::nodeBarrier()
{
   if ( _api != NULL )
   {
      _api->nodeBarrier();
   }
}

void Network::getNotify( unsigned int remoteNode, uint64_t remoteAddr )
{
   if ( _api != NULL )
   {
      _api->getNotify( remoteNode, remoteAddr );
   }
}

void Network::setMasterHostname( char *name )
{
    //  _masterHostname = std::string( name );
   if ( _masterHostname == NULL )
      _masterHostname = new char[ ::strlen( name ) + 1 ];
   ::memcpy(_masterHostname, name, ::strlen( name ) );
   _masterHostname[ ::strlen( name ) ] = '\0';
}

//const std::string & Network::getMasterHostname() const
const char * Network::getMasterHostname() const
{
   return _masterHostname; 
}


void Network::sendRequestPut( unsigned int dest, uint64_t origAddr, unsigned int dataDest, uint64_t dstAddr, size_t len )
{
   if ( _api != NULL )
         {
      //std::cerr << "[" << _nodeNum << "] => " << dest << " " << __FUNCTION__ << std::endl;
                  _api->sendRequestPut(dest, origAddr, dataDest, dstAddr, len);
         }

}

void Network::setMasterDirectory(Directory *dir)
{
 if ( _api != NULL) 
{
_api->setMasterDirectory( dir );
}
}

void Network::setPollingMinThd( unsigned int id)
{
	_pollingMinThd = id;
}
void Network::setPollingMaxThd( unsigned int id)
{
	_pollingMaxThd = id;
}
