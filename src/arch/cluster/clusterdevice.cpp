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


#include "clusterdevice.hpp"
#include "basethread.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "clusternode.hpp"
#include <iostream>

using namespace nanos;
using namespace nanos::ext;

void * ClusterDevice::allocate( size_t size, ProcessingElement *pe )
{
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   void *retAddr = NULL;

   retAddr = node->getAllocator().allocate( size );
   //fprintf(stderr, "[node %d] ALLOCATE %d at %d, ret %p\n", sys.getNetwork()->getNodeNum(), size, node->getClusterNodeNum(), retAddr );
   return retAddr;
}

void ClusterDevice::free( void *address, ProcessingElement *pe )
{
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );

   node->getAllocator().free( address );

   //unsigned int nodeId = node->getClusterNodeNum();
   //fprintf(stderr, "[node %d] FREE %p\n", nodeId, address);
}

void * ClusterDevice::realloc( void *address, size_t newSize, size_t oldSize, ProcessingElement *pe )
{
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );

   return node->getAllocator().reallocate( address, newSize );
}

bool ClusterDevice::copyDevToDev( void *addrSrc, size_t size, ProcessingElement *pe, ProcessingElement *peDst, void *addrDst )
{
   sys.getNetwork()->sendRequestPut( ((ClusterNode *) pe)->getClusterNodeNum(), (uint64_t )addrSrc, ((ClusterNode *) peDst)->getClusterNodeNum(), (uint64_t)addrDst, size );
   return true;
}

bool ClusterDevice::copyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size, ProcessingElement *pe )
{
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   //fprintf(stderr, "[node %d]->[%d] COPY IN ( remote=%p, <= local=0x%llx[%d], size=%d)\n", sys.getNetwork()->getNodeNum(), node->getClusterNodeNum(), localDst, remoteSrc, *((int *)remoteSrc), size);
   sys.getNetwork()->put( node->getClusterNodeNum(), ( uint64_t ) localDst, ( void * ) remoteSrc.getTag(), size );
   return true;
}

bool ClusterDevice::copyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size, ProcessingElement *pe )
{
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   //fprintf(stderr, "[node %d] COPY OUT from %d ( remote=%p, => local=0x%llx, size %d\n", sys.getNetwork()->getNodeNum(), node->getClusterNodeNum(), localSrc, remoteDst.getTag(), size);
   sys.getNetwork()->get( ( void * ) remoteDst.getTag(), node->getClusterNodeNum(), ( uint64_t ) localSrc, size );
   //fprintf(stderr, "[node %d]<-[%d] COPY OUT ( remote=%p, => local=0x%llx[%d], size %d\n", sys.getNetwork()->getNodeNum(),node->getClusterNodeNum(), localSrc, remoteDst, *((int *)remoteDst), size);
   return true;
}

void ClusterDevice::syncTransfer ( uint64_t address, ProcessingElement *pe )
{
}
