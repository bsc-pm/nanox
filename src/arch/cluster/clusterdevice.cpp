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

   //std::cerr << node->getNodeNum() << " Alloc of " << size << " ( " << (void *) size << " ) " << std::endl;
   retAddr = node->getAllocator().allocate( size );
   //std::cerr << node->getNodeNum() << " res is " << (void *) retAddr << std::endl;
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

void ClusterDevice::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, ProcessingElement *pe, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   //std::cerr << __FUNCTION__ << std::endl;
   sys.getNetwork()->put( node->getClusterNodeNum(),  devAddr, ( void * ) hostAddr, len, wdId, wd );
   ops->completeOp();
}
void ClusterDevice::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, ProcessingElement *pe, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   sys.getNetwork()->get( ( void * ) hostAddr, node->getClusterNodeNum(), devAddr, len );
   ops->completeOp();
}
void ClusterDevice::_copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, ProcessingElement *peDest, ProcessingElement *peOrig, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   //ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   sys.getNetwork()->sendRequestPut( ((ClusterNode *) peOrig)->getClusterNodeNum(), devOrigAddr, ((ClusterNode *) peDest)->getClusterNodeNum(), devDestAddr, len, wdId, wd );
   ops->completeOp();
}
void ClusterDevice::_copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t count, std::size_t ld, ProcessingElement *pe, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   char * hostAddrPtr = (char *) hostAddr;
   //
   NANOS_INSTRUMENT( InstrumentState inst2(NANOS_STRIDED_COPY_PACK); );
   //std::cerr << "getting addr-..."<< std::endl;
   char * packedAddr = (char *) _packer.give_pack( hostAddr, len, count );
   //std::cerr << "getting addr- done" <<std::endl;
   //std::cerr << __FUNCTION__ << " host addr is "<< (void *) hostAddr << " len is " << len << " count "<< count << " got packed data "<< ((void *) packedAddr) << " devAddr " << (void *) devAddr  << std::endl;
   if ( packedAddr != NULL) { 
      for ( unsigned int i = 0; i < count; i += 1 ) {
         //std::cerr << "::memcpy ( &packedAddr[ " << i << " * " << len << " ]="<< (void *) &packedAddr[ i * len ] << ", &hostAddrPtr[ " << i << " * " << ld << "]=" << (void *) &hostAddrPtr[ i * ld ] << ", " << len << " );" << std::endl;
         //std::cerr << "packed is " << (int ) packedAddr[ i * len ] << std::endl;
         //std::cerr << "hostaddr is " << (int ) hostAddrPtr[ i * ld ] << std::endl;
         ::memcpy( &packedAddr[ i * len ], &hostAddrPtr[ i * ld ], len );
      }
   } else { std::cerr << "copyInStrided ERROR!!! could not get a packet to gather data." << std::endl; }
   NANOS_INSTRUMENT( inst2.close(); );
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   //sys.getNetwork()->put( node->getClusterNodeNum(),  devAddr, ( void * ) hostAddr, len, wdId );
   //std::cerr <<"put Strided 1D" <<std::endl;
   sys.getNetwork()->putStrided1D( node->getClusterNodeNum(),  devAddr, ( void * ) hostAddr, packedAddr, len, count, ld, wdId, wd );
   _packer.free_pack( hostAddr, len, count );
   ops->completeOp();
}
void ClusterDevice::_copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t count, std::size_t ld, ProcessingElement *pe, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   char * hostAddrPtr = (char *) hostAddr;
   ClusterNode *node = dynamic_cast< ClusterNode * >( pe );
   //std::cerr << __FUNCTION__ << " for WD " << wdId << " hostAddr " << (void *)hostAddr << std::endl;

   std::size_t maxCount = ( ( len * count ) <= sys.getNetwork()->getMaxGetStridedLen() ) ?
      count : ( sys.getNetwork()->getMaxGetStridedLen() / len );
   //std::size_t maxRequestedSize = maxCount * len;

   //char * packedAddr = (char *) _packer.give_pack( hostAddr, len, count );
   //std::cerr  <<"get strided: host addr "<< (void *) hostAddr << " devAddr " << (void *) devAddr<< " get's " << maxCount<< std::endl;

   // FIXME: check if maxCount can copy more data on the last iteration
   if ( maxCount != count ) std::cerr <<"WARNING: maxCount("<< maxCount << ") != count(" << count <<")"<<std::endl;
   for ( unsigned int i = 0; i < count; i += maxCount ) {
      char * packedAddr = (char *) _packer.give_pack( hostAddr, len, maxCount );
      if ( packedAddr != NULL) { 
         GetRequestStrided *newreq = NEW GetRequestStrided( &hostAddrPtr[ i * ld ] , len, maxCount, ld, packedAddr, ops, &_packer );
   //std::cerr << "add request to thd " << myThread->getId() << std::endl; 
   myThread->_pendingRequests.insert( newreq );
         //volatile int req = 0;
         //volatile int *reqPtr = &req;
         ops->addOp();
         //std::cerr << "send request "<< ops;
         sys.getNetwork()->getStrided1D( packedAddr, node->getClusterNodeNum(), devAddr, devAddr + ( i * ld ), len, maxCount, ld, (volatile int *) newreq );
         //sys.getNetwork()->getStrided1D( packedAddr, node->getClusterNodeNum(), devAddr, devAddr + ( i * ld ), len, maxCount, ld, reqPtr );
         //while ( *reqPtr == 0 );// sys.getNetwork()->poll( 0 );
         //newreq->clear();
         //
         //NANOS_INSTRUMENT( InstrumentState inst2(NANOS_STRIDED_COPY_UNPACK); );
         //for ( unsigned int j = 0; j < count; j += 1 ) {
         //   ::memcpy( &hostAddrPtr[ ( j + i ) * ld ], &packedAddr[ j * len ], len );
         //}
         //NANOS_INSTRUMENT( inst2.close(); );
      } else { std::cerr << "copyOutStrdided ERROR!!! could not get a packet to gather data." << std::endl; }
   }

   //_packer.free_pack( hostAddr, len, count );
   //ops->completeOp();
}
void ClusterDevice::_copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, std::size_t count, std::size_t ld, ProcessingElement *peDest, ProcessingElement *peOrig, DeviceOps *ops, unsigned int wdId, WD *wd ) {
   //std::cerr << __FUNCTION__ << std::endl;
   //ClusterNode *node = dynamic_cast< ClusterNode * >( peOrig );
   sys.getNetwork()->sendRequestPutStrided1D( ((ClusterNode *) peOrig)->getClusterNodeNum(), devOrigAddr, ((ClusterNode *) peDest)->getClusterNodeNum(), devDestAddr, len, count, ld, wdId, wd );
   ops->completeOp();
}



void ClusterDevice::GetRequestStrided::clear() {
   //std::cerr <<"clear request "<< (void*) _ops << std::endl;
   NANOS_INSTRUMENT( InstrumentState inst2(NANOS_STRIDED_COPY_UNPACK); );
   for ( unsigned int j = 0; j < _count; j += 1 ) {
      ::memcpy( &_hostAddr[ j  * _ld ], &_recvAddr[ j * _size ], _size );
   }
   NANOS_INSTRUMENT( inst2.close(); );
   _packer->free_pack( (uint64_t) _hostAddr, _size, _count );
   _ops->completeOp();
   //std::cerr <<"clear request "<< (void*) _ops << std::endl;
}
