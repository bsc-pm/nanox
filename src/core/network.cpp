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

using namespace nanos;

Network::Network () : _numNodes( 0 ), _api( (NetworkAPI *) 0 ), _nodeNum( 0 ), _notify( NULL ),
                      _malloc_return( NULL ), _malloc_complete( NULL ), _masterHostname ( NULL ) {}
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

   _numNodes = numNodes;
   
   _notify = new volatile unsigned int[numNodes * sys.getNumPEs() ];
   _malloc_return = new void *[numNodes * sys.getNumPEs()];
   _malloc_complete = new bool[numNodes * sys.getNumPEs()];

   for (i = 0; i < numNodes * sys.getNumPEs(); i++)
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

void Network::poll()
{
//   ensure ( _api != NULL, "No network api loaded." );
   if (_api != NULL)
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

void Network::sendWorkMsg( unsigned int dest, void ( *work ) ( void * ), unsigned int dataSize, unsigned int wdId, unsigned int numPe, size_t argSize, void * arg )
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

         _api->sendWorkMsg( dest, work, dataSize, wdId, numPe, argSize, arg );
      }
      else
      {
         std::cerr << "tried to send work from a node != 0" << std::endl;
      }
   }
}

void Network::waitWorkCompletion(unsigned int dest, unsigned int numPe)
{
   //std::cerr << "waiting work from " << dest << " target pe " << numPe << std::endl;
   while (_notify[ dest * sys.getNumPEs() + numPe ] == 0)
   {
      poll();
      Scheduler::yield();
   }
   _notify[ dest * sys.getNumPEs() + numPe ] = 0;
   //std::cerr << "completed work from " << dest << std::endl;
}

void Network::sendWorkDoneMsg( unsigned int nodeNum, unsigned int numPe )
{
 //  ensure ( _api != NULL, "No network api loaded." );
   if ( _api != NULL )
   {
      NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
      NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) numPe) << 32 ) + _nodeNum; )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEventNkvs( NANOS_WD_REMOTE, id, 0, NULL, NULL, 0 ); )
      if ( _nodeNum != MASTER_NODE_NUM )
      {
         _api->sendWorkDoneMsg( nodeNum, numPe );
      }
   }
}

void Network::notifyWorkDone ( unsigned int nodeNum, unsigned int numPe )
{
   //std::cerr << "completed work from " << nodeNum << " : " << numPe << std::endl;
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
   NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) numPe) << 32 ) + nodeNum; )
   NANOS_INSTRUMENT ( instr->raiseClosePtPEventNkvs( NANOS_WD_REMOTE, id, 0, NULL, NULL, nodeNum ); )

   _notify[ nodeNum * sys.getNumPEs() + numPe ] = 1;
}

void Network::put ( unsigned int remoteNode, uint64_t remoteAddr, void *localAddr, size_t size )
{
   if ( _api != NULL )
      _api->put( remoteNode, remoteAddr, localAddr, size );
}

void Network::get ( void *localAddr, unsigned int remoteNode, uint64_t remoteAddr, size_t size )
{
   if ( _api != NULL )
      _api->get( localAddr, remoteNode, remoteAddr, size );
}

void * Network::malloc ( unsigned int remoteNode, size_t size, unsigned int id )
{
   void * result = NULL;

   if ( _api != NULL )
   {
      _api->malloc( remoteNode, size, id );

      while ( _malloc_complete[ remoteNode * sys.getNumPEs() + id ] == false )
      {
         poll();
      }

      result = _malloc_return[ remoteNode * sys.getNumPEs() + id ];
      _malloc_complete[ remoteNode * sys.getNumPEs() + id ] = false;
   }

   return result;
}

void Network::notifyMalloc( unsigned int remoteNode, void * addr, unsigned int id )
{
   _malloc_return[ remoteNode * sys.getNumPEs() + id ] = addr;
   _malloc_complete[ remoteNode * sys.getNumPEs() + id ] = true;
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
