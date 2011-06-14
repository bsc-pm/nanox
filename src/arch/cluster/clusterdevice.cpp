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


#include "clusterdevice_decl.hpp"
#include "basethread.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "clusternode_decl.hpp"
#include <iostream>

using namespace nanos;
using namespace nanos::ext;

ClusterDevice nanos::ext::Cluster( "SMP" );

void * ClusterDevice::allocate( size_t size, ProcessingElement *pe )
{
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   void *retAddr = NULL;

   retAddr = node->getAllocator().allocate( size );
   return retAddr;
}

void ClusterDevice::free( void *address, ProcessingElement *pe )
{
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );

   node->getAllocator().free( address );

   sys.getNetwork()->memFree( ((ClusterNode *) pe)->getClusterNodeNum(), address );
}

void * ClusterDevice::realloc( void *address, size_t newSize, size_t oldSize, ProcessingElement *pe )
{
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   void *retAddr = NULL;

   retAddr = node->getAllocator().allocate( newSize );
   
   sys.getNetwork()->memRealloc(((ClusterNode *) pe)->getClusterNodeNum(), address, oldSize, retAddr, newSize );
   return retAddr;
}

bool ClusterDevice::copyDevToDev( void *addrSrc, size_t size, ProcessingElement *pe, ProcessingElement *peDst, void *addrDst )
{
   sys.getNetwork()->sendRequestPut( ((ClusterNode *) pe)->getClusterNodeNum(), (uint64_t )addrSrc, ((ClusterNode *) peDst)->getClusterNodeNum(), (uint64_t)addrDst, size );
   return true;
}

bool ClusterDevice::copyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size, ProcessingElement *pe )
{
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   sys.getNetwork()->put( node->getClusterNodeNum(), ( uint64_t ) localDst, ( void * ) remoteSrc.getTag(), size );
   return true;
}

bool ClusterDevice::copyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size, ProcessingElement *pe )
{
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   sys.getNetwork()->get( ( void * ) remoteDst.getTag(), node->getClusterNodeNum(), ( uint64_t ) localSrc, size );
   return true;
}

void ClusterDevice::syncTransfer ( uint64_t address, ProcessingElement *pe )
{
}
