/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
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
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/


#include "clusterdevice_decl.hpp"
#include "basethread.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "network_decl.hpp"
#include "clusternode_decl.hpp"
#include "deviceops.hpp"
#include <iostream>

using namespace nanos;
using namespace nanos::ext;

ClusterDevice nanos::ext::Cluster( "Cluster" );


ClusterDevice::ClusterDevice ( const char *n ) : Device ( n ) {
}

ClusterDevice::ClusterDevice ( const ClusterDevice &arch ) : Device ( arch ) {
}

ClusterDevice::~ClusterDevice() {
}

void * ClusterDevice::memAllocate( size_t size, SeparateMemoryAddressSpace &mem, WorkDescriptor const *wd, unsigned int copyIdx) {
   void *retAddr = NULL;

   SimpleAllocator *allocator = (SimpleAllocator *) mem.getSpecificData();
   allocator->lock();
   retAddr = allocator->allocate( size );
   allocator->unlock();
   return retAddr;
}

void ClusterDevice::memFree( uint64_t addr, SeparateMemoryAddressSpace &mem ) {
   SimpleAllocator *allocator = (SimpleAllocator *) mem.getSpecificData();
   allocator->lock();
   allocator->free( (void *) addr );
   allocator->unlock();
}

void ClusterDevice::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) {
   ops->addOp();
   sys.getNetwork()->put( mem.getNodeNumber(),  devAddr, ( void * ) hostAddr, len, wd->getId(), wd, hostObject, hostRegionId );
   ops->completeOp();
}

void ClusterDevice::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) {

   char *recvAddr = NULL;
   do { 
      recvAddr = (char *) sys.getNetwork()->allocateReceiveMemory( len );
      if ( !recvAddr ) {
         myThread->processTransfers();
      }
   } while ( recvAddr == NULL );

   GetRequest *newreq = NEW GetRequest( (char *) hostAddr, len, recvAddr, ops );
   myThread->_pendingRequests.insert( newreq );

   ops->addOp();
   sys.getNetwork()->get( ( void * ) recvAddr, mem.getNodeNumber(), devAddr, len, newreq, hostObject, hostRegionId );
}

bool ClusterDevice::_copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) {
   ops->addOp();
   sys.getNetwork()->sendRequestPut( memOrig.getNodeNumber(), devOrigAddr, memDest.getNodeNumber(), devDestAddr, len, wd->getId(), wd, hostObject, hostRegionId );
   ops->completeOp();
   return true;
}

void ClusterDevice::_copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) {
   char *hostAddrPtr = (char *) hostAddr;
   char *packedAddr = NULL;
   ops->addOp();
   //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_STRIDED_COPY_PACK); );
      //*myThread->_file << "Allocate " << len * count << " to pack data (len=" << len << " count=" << count << " ld=" <<ld << ")"<< std::endl;
   do {
      packedAddr = (char *) _packer.give_pack( hostAddr, len, count );
      if (!packedAddr ) {
         myThread->processTransfers();
      }
   } while ( packedAddr == NULL );
      //*myThread->_file << "Got address " << (void *)packedAddr << std::endl;

   if ( packedAddr != NULL) { 
      for ( unsigned int i = 0; i < count; i += 1 ) {
         ::memcpy( &packedAddr[ i * len ], &hostAddrPtr[ i * ld ], len );
      }
   } else { std::cerr << "copyInStrided ERROR!!! could not get a packet to gather data." << std::endl; }
   //NANOS_INSTRUMENT( inst2.close(); );
   sys.getNetwork()->putStrided1D( mem.getNodeNumber(),  devAddr, ( void * ) hostAddr, packedAddr, len, count, ld, wd->getId(), wd, hostObject, hostRegionId );
   if ( _packer.free_pack( hostAddr, len, count, packedAddr ) == false ) {
      *myThread->_file << "Error freeing pack after sending copyIn to node " << mem.getNodeNumber() << " HostAddr " << (void *) hostAddr << " wd: " << wd->getId() << " region: " << (void *) hostObject << ":" << hostRegionId << std::endl;
   }
   ops->completeOp();
}

void ClusterDevice::_copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) {
   char * hostAddrPtr = (char *) hostAddr;
   //std::cerr << "ClusterDevice::_copyOutStrided1D with count " << count << " and len " << len << " sys.getNetwork()->getMaxGetStridedLen() is " << sys.getNetwork()->getMaxGetStridedLen()<< std::endl;
   std::size_t maxCount = ( ( len * count ) <= sys.getNetwork()->getMaxGetStridedLen() ) ? count : ( sys.getNetwork()->getMaxGetStridedLen() / len );

   //if ( maxCount != count ) std::cerr <<"WARNING: maxCount("<< maxCount << ") != count(" << count <<") MaxGetStridedLen="<< sys.getNetwork()->getMaxGetStridedLen()<<std::endl;
   if ( maxCount ) {
      for ( unsigned int i = 0; i < count; i += maxCount ) {
         unsigned int thisCount = ( i + maxCount > count ) ? count - i : maxCount; 
         char * packedAddr = NULL;
         do {
            packedAddr = (char *) _packer.give_pack( hostAddr, len, thisCount );
            if (!packedAddr ) {
               myThread->processTransfers();
            }
         } while ( packedAddr == NULL );

         if ( packedAddr != NULL) { 
            GetRequestStrided *newreq = NEW GetRequestStrided( &hostAddrPtr[ i * ld ] , len, thisCount, ld, packedAddr, ops, &_packer );
            myThread->_pendingRequests.insert( newreq );
            ops->addOp();
            sys.getNetwork()->getStrided1D( packedAddr, mem.getNodeNumber(), devAddr, devAddr + ( i * ld ), len, thisCount, ld, newreq, hostObject, hostRegionId );
         } else {
            std::cerr << "copyOutStrdided ERROR!!! could not get a packet to gather data." << std::endl;
         }
      }
   } else {
      /* len > sys.getNetwork()->getMaxGetStridedLen()
       * use non strided gets
       */
      for ( unsigned int i = 0; i < count; i += 1) {
         for ( std::size_t current_line_sent = 0; current_line_sent < len; current_line_sent += sys.getNetwork()->getMaxGetStridedLen() ) {
            std::size_t current_len = ( current_line_sent + sys.getNetwork()->getMaxGetStridedLen() ) > len ? ( len - current_line_sent ) : sys.getNetwork()->getMaxGetStridedLen();
            std::size_t current_offset = i * ld + current_line_sent;
            //std::cerr << "copyOut offset: " << current_offset << " len " << current_len << std::endl;
            this->_copyOut( hostAddr + current_offset , devAddr + current_offset, current_len, mem, ops, wd, hostObject, hostRegionId );
         }
      }
   }
}

bool ClusterDevice::_copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId ) {
   ops->addOp();
   sys.getNetwork()->sendRequestPutStrided1D( memOrig.getNodeNumber(), devOrigAddr, memDest.getNodeNumber(), devDestAddr, len, count, ld, wd->getId(), wd, hostObject, hostRegionId );
   ops->completeOp();
   return true;
}

void ClusterDevice::_canAllocate( SeparateMemoryAddressSpace &mem, std::size_t *sizes, unsigned int numChunks, std::size_t *remainingSizes ) {
   SimpleAllocator *allocator = (SimpleAllocator *) mem.getSpecificData();
   allocator->canAllocate( sizes, numChunks, remainingSizes );
}
void ClusterDevice::_getFreeMemoryChunksList( SeparateMemoryAddressSpace &mem, SimpleAllocator::ChunkList &list ) {
   SimpleAllocator *allocator = (SimpleAllocator *) mem.getSpecificData();
   allocator->getFreeChunksList( list );
}

std::size_t ClusterDevice::getMemCapacity( SeparateMemoryAddressSpace &mem ) {
   SimpleAllocator *allocator = (SimpleAllocator *) mem.getSpecificData();
   return allocator->getCapacity();
}
