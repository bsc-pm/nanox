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
#include "network_decl.hpp"
#include "clusternode_decl.hpp"
#include "deviceops.hpp"
#include <iostream>

using namespace nanos;
using namespace nanos::ext;

ClusterDevice nanos::ext::Cluster( "SMP" );

ClusterDevice::GetRequest::GetRequest( char* hostAddr, std::size_t size, char *recvAddr, DeviceOps *ops ) : _complete(0),
   _hostAddr( hostAddr ), _size( size ), _recvAddr( recvAddr ), _ops( ops ) {
}

ClusterDevice::GetRequest::~GetRequest() {
}

void ClusterDevice::GetRequest::complete() {
   _complete = 1;
}

bool ClusterDevice::GetRequest::isCompleted() const {
   return _complete == 1;
}

void ClusterDevice::GetRequest::clear() {
   ::memcpy( _hostAddr, _recvAddr, _size );
   sys.getNetwork()->freeReceiveMemory( _recvAddr );
   _ops->completeOp();
}

ClusterDevice::GetRequestStrided::GetRequestStrided( char* hostAddr, std::size_t size, std::size_t count, std::size_t ld, char *recvAddr, DeviceOps *ops, Packer *packer ) :
   GetRequest( hostAddr, size, recvAddr, ops ), _count( count ), _ld( ld ), _packer( packer ) {
}

ClusterDevice::GetRequestStrided::~GetRequestStrided() {
}

void ClusterDevice::GetRequestStrided::clear() {
   NANOS_INSTRUMENT( InstrumentState inst2(NANOS_STRIDED_COPY_UNPACK); );
   for ( unsigned int j = 0; j < _count; j += 1 ) {
      ::memcpy( &_hostAddr[ j  * _ld ], &_recvAddr[ j * _size ], _size );
   }
   NANOS_INSTRUMENT( inst2.close(); );
   _packer->free_pack( (uint64_t) _hostAddr, _size, _count, _recvAddr );
   _ops->completeOp();
}

ClusterDevice::ClusterDevice ( const char *n ) : Device ( n ) {
}

ClusterDevice::ClusterDevice ( const ClusterDevice &arch ) : Device ( arch ) {
}

ClusterDevice::~ClusterDevice() {
}

void * ClusterDevice::memAllocate( size_t size, ProcessingElement &pe ) const
{
   ClusterNode &node = dynamic_cast< ClusterNode & >( pe );
   void *retAddr = NULL;

   node.getAllocator().lock();
   retAddr = node.getAllocator().allocate( size );
   node.getAllocator().unlock();
   return retAddr;
}

void ClusterDevice::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, ProcessingElement &pe, DeviceOps *ops, WD const &wd ) const {
   ClusterNode const &node = dynamic_cast< ClusterNode const & >( pe );
   ops->addOp();
   sys.getNetwork()->put( node.getClusterNodeNum(),  devAddr, ( void * ) hostAddr, len, wd.getId(), wd );
   ops->completeOp();
}

void ClusterDevice::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, ProcessingElement &pe, DeviceOps *ops, WD const &wd ) const {
   ClusterNode const &node = dynamic_cast< ClusterNode const & >( pe );

   char *recvAddr = (char *) sys.getNetwork()->allocateReceiveMemory( len );

   GetRequest *newreq = NEW GetRequest( (char *) hostAddr, len, recvAddr, ops );
   myThread->_pendingRequests.insert( newreq );

   ops->addOp();
   sys.getNetwork()->get( ( void * ) recvAddr, node.getClusterNodeNum(), devAddr, len, (volatile int *) newreq );
}

void ClusterDevice::_copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, ProcessingElement &peDest, ProcessingElement &peOrig, DeviceOps *ops, WD const &wd ) const {
   ops->addOp();
   sys.getNetwork()->sendRequestPut( ((ClusterNode const &) peOrig).getClusterNodeNum(), devOrigAddr, ((ClusterNode const &) peDest).getClusterNodeNum(), devDestAddr, len, wd.getId(), wd );
   ops->completeOp();
}

void ClusterDevice::_copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t count, std::size_t ld, ProcessingElement const &pe, DeviceOps *ops, WD const &wd ) {
   char * hostAddrPtr = (char *) hostAddr;
   ops->addOp();
   NANOS_INSTRUMENT( InstrumentState inst2(NANOS_STRIDED_COPY_PACK); );
   char * packedAddr = (char *) _packer.give_pack( hostAddr, len, count );
   if ( packedAddr != NULL) { 
      for ( unsigned int i = 0; i < count; i += 1 ) {
         ::memcpy( &packedAddr[ i * len ], &hostAddrPtr[ i * ld ], len );
      }
   } else { std::cerr << "copyInStrided ERROR!!! could not get a packet to gather data." << std::endl; }
   NANOS_INSTRUMENT( inst2.close(); );
   ClusterNode const &node = dynamic_cast< ClusterNode const & >( pe );
   sys.getNetwork()->putStrided1D( node.getClusterNodeNum(),  devAddr, ( void * ) hostAddr, packedAddr, len, count, ld, wd.getId(), wd );
   _packer.free_pack( hostAddr, len, count, packedAddr );
   ops->completeOp();
}

void ClusterDevice::_copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t count, std::size_t ld, ProcessingElement const &pe, DeviceOps *ops, WD const &wd ) {
   char * hostAddrPtr = (char *) hostAddr;
   ClusterNode const &node = dynamic_cast< ClusterNode const & >( pe );

   std::size_t maxCount = ( ( len * count ) <= sys.getNetwork()->getMaxGetStridedLen() ) ?
      count : ( sys.getNetwork()->getMaxGetStridedLen() / len );

   // FIXME: check if maxCount can copy more data on the last iteration
   if ( maxCount != count ) std::cerr <<"WARNING: maxCount("<< maxCount << ") != count(" << count <<") MaxGetStridedLen="<< sys.getNetwork()->getMaxGetStridedLen()<<std::endl;
   for ( unsigned int i = 0; i < count; i += maxCount ) {
      char * packedAddr = (char *) _packer.give_pack( hostAddr, len, maxCount );
      if ( packedAddr != NULL) { 
         GetRequestStrided *newreq = NEW GetRequestStrided( &hostAddrPtr[ i * ld ] , len, maxCount, ld, packedAddr, ops, &_packer );
         myThread->_pendingRequests.insert( newreq );
         ops->addOp();
         sys.getNetwork()->getStrided1D( packedAddr, node.getClusterNodeNum(), devAddr, devAddr + ( i * ld ), len, maxCount, ld, (volatile int *) newreq );
      } else {
         std::cerr << "copyOutStrdided ERROR!!! could not get a packet to gather data." << std::endl;
      }
   }
}

void ClusterDevice::_copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, std::size_t count, std::size_t ld, ProcessingElement const &peDest, ProcessingElement const &peOrig, DeviceOps *ops, WD const &wd ) const {
   ops->addOp();
   sys.getNetwork()->sendRequestPutStrided1D( ((ClusterNode &) peOrig).getClusterNodeNum(), devOrigAddr, ((ClusterNode &) peDest).getClusterNodeNum(), devDestAddr, len, count, ld, wd.getId(), wd );
   ops->completeOp();
}

