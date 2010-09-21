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


#include <stdio.h>
#include "network.hpp"

using namespace nanos;

Network::Network () : _numNodes( 0 ), _api( (NetworkAPI *) 0 ), _nodeNum( 0 ), _notify( NULL ), _malloc_return( NULL ), _malloc_complete( NULL ) {}
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
   
   _notify = new unsigned int[numNodes];
   _malloc_return = new void *[numNodes];
   _malloc_complete = new bool[numNodes];

   for (i = 0; i < numNodes; i++)
   {
      _notify[ i ] = 0;
      _malloc_complete[ i ] = false;
   }
}

unsigned int Network::getNumNodes ()
{
   return _numNodes;
}

void Network::setNodeNum ( unsigned int nodeNum )
{
   _nodeNum = nodeNum;
}

unsigned int Network::getNodeNum ()
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
//   ensure ( _api != NULL, "No network api loaded." );
   if ( _api != NULL )
      _api->finalize();
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

void Network::sendWorkMsg( unsigned int dest, void ( *work ) ( void * ), unsigned int arg0, size_t argSize, void * arg )
{
 //  ensure ( _api != NULL, "No network api loaded." );
   if ( _api != NULL )
   {
      if (work == NULL)
      {
         fprintf(stderr, "ERROR\n");
      }
      if ( _nodeNum == MASTER_NODE_NUM )
      {
         _api->sendWorkMsg( dest, work, arg0, argSize, arg );

         while (_notify[dest] == 0)
            poll();
         _notify[dest] = 0;
      }
      else
      {
         fprintf(stderr, "tried to send work from a node != 0\n");
      }
   }
}

void Network::sendWorkDoneMsg( unsigned int nodeNum )
{
 //  ensure ( _api != NULL, "No network api loaded." );
   if ( _api != NULL )
   {
      if ( _nodeNum != MASTER_NODE_NUM )
      {
         _api->sendWorkDoneMsg( nodeNum );
      }
   }
}

void Network::notifyWorkDone ( unsigned int nodeNum )
{
   _notify[nodeNum] = 1;
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

void * Network::malloc ( unsigned int remoteNode, size_t size )
{
   void * result = NULL;

   if ( _api != NULL )
   {
      _api->malloc( remoteNode, size );

      while ( _malloc_complete[ remoteNode ] == false )
         poll();

      result = _malloc_return[ remoteNode ];
      _malloc_complete[ remoteNode ] = false;
   }

   return result;
}

void Network::notifyMalloc( unsigned int remoteNode, void * addr )
{
   _malloc_return[ remoteNode ] = addr;
   _malloc_complete[ remoteNode ] = true;
}
