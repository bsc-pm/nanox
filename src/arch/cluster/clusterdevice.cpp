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
#include "clusternode.hpp"
#include "basethread.hpp"
#include "debug.hpp"
#include "system.hpp"

using namespace nanos;
using namespace ext;

//std::vector< ClusterDevice::SegmentMap > ClusterDevice::_allocatedChunks;
//std::vector< ClusterDevice::SegmentMap > ClusterDevice::_freeChunks;
unsigned int ClusterDevice::_numSegments = 0;
void ** ClusterDevice::_segmentAddrList = NULL;
size_t * ClusterDevice::_segmentLenList = NULL;

unsigned int ClusterDevice::_extraPEsCount = 0;

void * ClusterDevice::allocate( size_t size )
{
   ClusterNode *node = ( ClusterNode * ) myThread->runningOn();
   void *retAddr = NULL;

   retAddr = node->getAllocator().allocate( size );
   /*
   unsigned int nodeId = node->getClusterNodeNum();
   SegmentMap &nodeFreeChunks = _freeChunks[ nodeId ];
   SegmentMap &nodeAllocatedChunks = _allocatedChunks[ nodeId ];
   SegmentMap::iterator mapIter = nodeFreeChunks.begin();
   //fprintf(stderr, "[node %d] ALLOCATE %d at %d, ret %p\n", sys.getNetwork()->getNodeNum(), size, node->getClusterNodeNum(), addr );

   //fprintf(stderr, "looking for chunk of %d bytes, first is %d bytes.", size, mapIter->second );
   while( mapIter != nodeFreeChunks.end() && mapIter->second < size )
   {
      mapIter++;
   }

   if ( mapIter != nodeFreeChunks.end() ) {

      uintptr_t targetAddr = mapIter->first;
      size_t chunkSize = mapIter->second;

      nodeFreeChunks.erase( mapIter );

      //add the chunk with the new size (previous size - requested size)
      nodeFreeChunks[ targetAddr + size ] = chunkSize - size ;
      nodeAllocatedChunks[ targetAddr ] = size;

      retAddr = ( void * ) targetAddr;
   }
   else {
      fprintf(stderr, "Could not get a chunk of %d bytes at node %d\n", size, nodeId);
   }
   */
   
   return retAddr;
}

void ClusterDevice::free( void *address )
{
   ClusterNode *node = ( ClusterNode * ) myThread->runningOn();

   node->getAllocator().free( address );
   unsigned int nodeId = node->getClusterNodeNum();
   /*
   SegmentMap &nodeFreeChunks = _freeChunks[ nodeId ];
   SegmentMap &nodeAllocatedChunks = _allocatedChunks[ nodeId ];
   
   SegmentMap::iterator mapIter = nodeAllocatedChunks.find( ( uintptr_t ) address );
   size_t size = mapIter->second;

   nodeAllocatedChunks.erase( mapIter );

   if ( nodeFreeChunks.size() > 0 ) {
      mapIter = nodeFreeChunks.lower_bound( ( uintptr_t ) address );

      //case where address is the highest key, check if it can be merged with the previous chunk
      if ( mapIter == nodeFreeChunks.end() ) {
         mapIter--;
         if ( mapIter->first + mapIter->second == ( uintptr_t ) address ) {
            mapIter->second += size;
         }
      }
      //address is not the hightest key, check if it can be merged with the previous and next chunks
      else if ( nodeFreeChunks.key_comp()( ( uintptr_t ) address, mapIter->first ) ) {
         size_t totalSize = size;

         //check next chunk
         if ( mapIter->first == ( ( uintptr_t ) address ) + size ) {
            totalSize += mapIter->second;
            nodeFreeChunks.erase( mapIter );
            mapIter = nodeFreeChunks.insert( mapIter, SegmentMap::value_type( ( uintptr_t ) address, totalSize ) );
         }

         //check previous chunk
         mapIter--;
         if ( mapIter->first + mapIter->second == ( uintptr_t ) address ) {
            //if totalSize > size then the next chunk was merged
            if ( totalSize > size ) {
               mapIter->second += totalSize;
               mapIter++;
               nodeFreeChunks.erase( mapIter );
            }
            else {
               mapIter->second += size;
            }
         }
      }
      //duplicate key, error
      else {
         fprintf(stderr, "Duplicate entry in node segment map, node %d, addr %p\n.", nodeId, address);
      }
   }
   else {
      nodeFreeChunks[ ( uintptr_t ) address ] = size;
   }
   */


   
   fprintf(stderr, "[node %d] FREE %p\n", nodeId, address);
}

void ClusterDevice::copyIn( void *localDst, uint64_t remoteSrc, size_t size )
{
   ClusterNode *node = (ClusterNode *) myThread->runningOn();
   //fprintf(stderr, "[node %d] COPY IN ( remote=%p, <= local=0x%llx[%d], size=%d)\n", sys.getNetwork()->getNodeNum(), localDst, remoteSrc, *((int *)remoteSrc), size);
   sys.getNetwork()->put( node->getClusterNodeNum(), ( uint64_t ) localDst, ( void * ) remoteSrc, size );
}

void ClusterDevice::copyOut( uint64_t remoteDst, void *localSrc, size_t size )
{
   ClusterNode *node = (ClusterNode *) myThread->runningOn();
   //fprintf(stderr, "[node %d] COPY OUT ( remote=%p, => local=0x%llx[%d], size %d\n", sys.getNetwork()->getNodeNum(), localSrc, remoteDst, *((int *)remoteDst), size);
   sys.getNetwork()->get( ( void * ) remoteDst, node->getClusterNodeNum(), ( uint64_t ) localSrc, size );
   //fprintf(stderr, "[node %d] COPY OUT ( remote=%p, => local=0x%llx[%d], size %d\n", sys.getNetwork()->getNodeNum(), localSrc, remoteDst, *((int *)remoteDst), size);
}

