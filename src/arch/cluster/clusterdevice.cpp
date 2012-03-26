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
   //retAddr = sys.getNetwork()->malloc( node->getClusterNodeNum(), size );
   return retAddr;
}

void ClusterDevice::free( void *address, ProcessingElement *pe )
{
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );

   node->getAllocator().free( address );

   message("WARNING: calling ClusterDevice::free");

   //sys.getNetwork()->memFree( node->getClusterNodeNum(), address );
}

void * ClusterDevice::realloc( void *address, size_t newSize, size_t oldSize, ProcessingElement *pe )
{
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   void *retAddr = NULL;

   retAddr = node->getAllocator().allocate( newSize );
   
   //sys.getNetwork()->memRealloc(node->getClusterNodeNum(), address, oldSize, retAddr, newSize );
   return retAddr;
}

bool ClusterDevice::copyDevToDev( void * addrDst, CopyDescriptor &dstCd, void * addrSrc, std::size_t size, ProcessingElement *peDst, ProcessingElement *peSrc )
{
   std::cerr<<"ERROR; OLD CACHE" << std::endl;
   //sys.getNetwork()->sendRequestPut( ((ClusterNode *) peSrc)->getClusterNodeNum(), (uint64_t )addrSrc, ((ClusterNode *) peDst)->getClusterNodeNum(), (uint64_t)addrDst, size );
   return true;
}

bool ClusterDevice::copyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size, ProcessingElement *pe )
{
   //ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   //sys.getNetwork()->put( node->getClusterNodeNum(), ( uint64_t ) localDst, ( void * ) remoteSrc.getTag(), size );
   return true;
}

bool ClusterDevice::copyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size, ProcessingElement *pe )
{
   //ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   //sys.getNetwork()->get( ( void * ) remoteDst.getTag(), node->getClusterNodeNum(), ( uint64_t ) localSrc, size );
   return true;
}

void ClusterDevice::syncTransfer ( uint64_t address, ProcessingElement *pe )
{
}

void ClusterDevice::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, ProcessingElement *pe, DeviceOps *ops, unsigned int wdId ) {
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   sys.getNetwork()->put( node->getClusterNodeNum(),  devAddr, ( void * ) hostAddr, len, wdId );
   ops->completeOp();
}
void ClusterDevice::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, ProcessingElement *pe, DeviceOps *ops ) {
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   sys.getNetwork()->get( ( void * ) hostAddr, node->getClusterNodeNum(), devAddr, len );
   ops->completeOp();
}
void ClusterDevice::_copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, ProcessingElement *peDest, ProcessingElement *peOrig, DeviceOps *ops, unsigned int wdId ) {
   //ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   sys.getNetwork()->sendRequestPut( ((ClusterNode *) peOrig)->getClusterNodeNum(), devOrigAddr, ((ClusterNode *) peDest)->getClusterNodeNum(), devDestAddr, len, wdId );
   ops->completeOp();
}
