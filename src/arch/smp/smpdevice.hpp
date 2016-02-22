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

#ifndef _SMP_DEVICE_DECL
#define _SMP_DEVICE_DECL

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "workdescriptor_decl.hpp"
#include "processingelement_fwd.hpp"
#include "copydescriptor.hpp"
#include "system_decl.hpp"
#include "smptransferqueue.hpp"

namespace nanos
{

SMPDevice::SMPDevice ( const char *n ) : Device ( n ), _transferQueue() {}
SMPDevice::SMPDevice ( const SMPDevice &arch ) : Device ( arch ), _transferQueue() {}

/*! \brief SMPDevice destructor
 */
SMPDevice::~SMPDevice() {};

void *SMPDevice::memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem, WorkDescriptor const &wd, unsigned int copyIdx ) {
   void *retAddr = NULL;

   SimpleAllocator *sallocator = (SimpleAllocator *) mem.getSpecificData();
   sallocator->lock();
   retAddr = sallocator->allocate( size );
   sallocator->unlock();
   return retAddr;
}

void SMPDevice::memFree( uint64_t addr, SeparateMemoryAddressSpace &mem ) {
   SimpleAllocator *sallocator = (SimpleAllocator *) mem.getSpecificData();
   sallocator->lock();
   sallocator->free( (void *) addr );
   sallocator->unlock();
}

void SMPDevice::_canAllocate( SeparateMemoryAddressSpace &mem, std::size_t *sizes, unsigned int numChunks, std::size_t *remainingSizes ) {
   SimpleAllocator *sallocator = (SimpleAllocator *) mem.getSpecificData();
   sallocator->canAllocate( sizes, numChunks, remainingSizes );
}

std::size_t SMPDevice::getMemCapacity( SeparateMemoryAddressSpace &mem ) {
   SimpleAllocator *sallocator = (SimpleAllocator *) mem.getSpecificData();
   return sallocator->getCapacity();
}

void SMPDevice::_copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WorkDescriptor const &wd, void *hostObject, reg_t hostRegionId ) {
   if ( sys.getSMPPlugin()->asyncTransfersEnabled() ) {
      _transferQueue.addTransfer( ops, ((char *) devAddr), ((char *) hostAddr), len, 1, 0, true );
   } else {
      ops->addOp();
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t key = ID->getEventKey("cache-copy-in"); )
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( key, (nanos_event_value_t) len ); )
      ::memcpy( (void *) devAddr, (void *) hostAddr, len );
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( key, (nanos_event_value_t) 0 ); )
      ops->completeOp();
   }
}

void SMPDevice::_copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WorkDescriptor const &wd, void *hostObject, reg_t hostRegionId ) {
   if ( sys.getSMPPlugin()->asyncTransfersEnabled() ) {
      _transferQueue.addTransfer( ops, ((char *) hostAddr), ((char *) devAddr), len, 1, 0, true );
   } else {
      ops->addOp();
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t key = ID->getEventKey("cache-copy-out"); )
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( key, (nanos_event_value_t) len ); )
      ::memcpy( (void *) hostAddr, (void *) devAddr, len );
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( key, (nanos_event_value_t) 0 ); )
      ops->completeOp();
   }
}

bool SMPDevice::_copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memorig, DeviceOps *ops, WorkDescriptor const &wd, void *hostObject, reg_t hostRegionId ) {
   if ( sys.getSMPPlugin()->asyncTransfersEnabled() ) {
      _transferQueue.addTransfer( ops, ((char *) devDestAddr), ((char *) devOrigAddr), len, 1, 0, true );
   } else {
      ops->addOp();
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t key = ID->getEventKey("cache-copy-in"); )
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( key, (nanos_event_value_t) len ); )
      ::memcpy( (void *) devDestAddr, (void *) devOrigAddr, len );
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( key, (nanos_event_value_t) 0 ); )
      ops->completeOp();
   }
   return true;
}

void SMPDevice::_copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WorkDescriptor const &wd, void *hostObject, reg_t hostRegionId ) {
   if ( sys.getSMPPlugin()->asyncTransfersEnabled() ) {
      _transferQueue.addTransfer( ops, ((char *) devAddr), ((char *) hostAddr), len, numChunks, ld, true );
   } else {
      ops->addOp();
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t key = ID->getEventKey("cache-copy-in"); )
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( key, (nanos_event_value_t) 2 ); )
      for ( std::size_t count = 0; count < numChunks; count += 1) {
         ::memcpy( ((char *) devAddr) + count * ld, ((char *) hostAddr) + count * ld, len );
      }
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( key, (nanos_event_value_t) 0 ); )
      ops->completeOp();
   }
}

void SMPDevice::_copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WorkDescriptor const &wd, void *hostObject, reg_t hostRegionId ) {
   if ( sys.getSMPPlugin()->asyncTransfersEnabled() ) {
      _transferQueue.addTransfer( ops, ((char *) hostAddr), ((char *) devAddr), len, numChunks, ld, false );
   } else {
      ops->addOp();
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t key = ID->getEventKey("cache-copy-out"); )
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( key, (nanos_event_value_t) 2 ); )
      for ( std::size_t count = 0; count < numChunks; count += 1) {
         ::memcpy( ((char *) hostAddr) + count * ld, ((char *) devAddr) + count * ld, len );
      }
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( key, (nanos_event_value_t) 0 ); )
      ops->completeOp();
   }
}

bool SMPDevice::_copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, WorkDescriptor const &wd, void *hostObject, reg_t hostRegionId ) {
   if ( sys.getSMPPlugin()->asyncTransfersEnabled() ) {
      _transferQueue.addTransfer( ops, ((char *) devDestAddr), ((char *) devOrigAddr), len, numChunks, ld, true );
   } else {
      ops->addOp();
      NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
      NANOS_INSTRUMENT ( static nanos_event_key_t key = ID->getEventKey("cache-copy-in"); )
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( key, (nanos_event_value_t) 2 ); )
      for ( std::size_t count = 0; count < numChunks; count += 1) {
         ::memcpy( ((char *) devDestAddr) + count * ld, ((char *) devOrigAddr) + count * ld, len );
      }
      NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( key, (nanos_event_value_t) 0 ); )
      ops->completeOp();
   }
   return true;
}

void SMPDevice::_getFreeMemoryChunksList( SeparateMemoryAddressSpace &mem, SimpleAllocator::ChunkList &list ) {
   SimpleAllocator *sallocator = (SimpleAllocator *) mem.getSpecificData();
   sallocator->getFreeChunksList( list );
}

void SMPDevice::tryExecuteTransfer() {
   _transferQueue.tryExecuteOne();
}
}

#endif

