/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

#ifndef _SMP_DEVICE
#define _SMP_DEVICE

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "workdescriptor_decl.hpp"
#include "processingelement_fwd.hpp"
#include "copydescriptor.hpp"

namespace nanos
{

  /* \brief Device specialization for SMP architecture
   * provides functions to allocate and copy data in the device
   */
   class SMPDevice : public Device
   {
      public:
         /*! \brief SMPDevice constructor
          */
         SMPDevice ( const char *n ) : Device ( n ) {}

         /*! \brief SMPDevice copy constructor
          */
         SMPDevice ( const SMPDevice &arch ) : Device ( arch ) {}

         /*! \brief SMPDevice destructor
          */
         ~SMPDevice() {};

         virtual void *memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem, WorkDescriptor const &wd, unsigned int copyIdx ) {
            void *retAddr = NULL;

            SimpleAllocator *sallocator = (SimpleAllocator *) mem.getSpecificData();
            sallocator->lock();
            retAddr = sallocator->allocate( size );
            sallocator->unlock();
            return retAddr;
         }

         virtual void memFree( uint64_t addr, SeparateMemoryAddressSpace &mem ) {
            SimpleAllocator *sallocator = (SimpleAllocator *) mem.getSpecificData();
            sallocator->lock();
            sallocator->free( (void *) addr );
            sallocator->unlock();
         }

         virtual void _canAllocate( SeparateMemoryAddressSpace const &mem, std::size_t *sizes, unsigned int numChunks, std::size_t *remainingSizes ) const {
            SimpleAllocator *sallocator = (SimpleAllocator *) mem.getSpecificData();
            sallocator->canAllocate( sizes, numChunks, remainingSizes );
         }

         virtual std::size_t getMemCapacity( SeparateMemoryAddressSpace const &mem ) const {
            SimpleAllocator *sallocator = (SimpleAllocator *) mem.getSpecificData();
            return sallocator->getCapacity();
         }

         virtual void _copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, Functor *f, WorkDescriptor const &wd, void *hostObject, reg_t hostRegionId ) const {
            ::memcpy( (void *) devAddr, (void *) hostAddr, len );
         }

         virtual void _copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, Functor *f, WorkDescriptor const &wd, void *hostObject, reg_t hostRegionId ) const {
            ::memcpy( (void *) hostAddr, (void *) devAddr, len );
         }

         virtual bool _copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memorig, DeviceOps *ops, Functor *f, WorkDescriptor const &wd, void *hostObject, reg_t hostRegionId ) const {
            ::memcpy( (void *) devDestAddr, (void *) devOrigAddr, len );
            return true;
         }

         virtual void _copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace const &mem, DeviceOps *ops, Functor *f, WorkDescriptor const &wd, void *hostObject, reg_t hostRegionId ) {
            for ( std::size_t count = 0; count < numChunks; count += 1) {
               ::memcpy( ((char *) devAddr) + count * ld, ((char *) hostAddr) + count * ld, len );
            }
         }

         virtual void _copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, Functor *f, WorkDescriptor const &wd, void *hostObject, reg_t hostRegionId ) {
            for ( std::size_t count = 0; count < numChunks; count += 1) {
               ::memcpy( ((char *) hostAddr) + count * ld, ((char *) devAddr) + count * ld, len );
            }
         }

         virtual bool _copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace const &memDest, SeparateMemoryAddressSpace const &memOrig, DeviceOps *ops, Functor *f, WorkDescriptor const &wd, void *hostObject, reg_t hostRegionId ) const {
            for ( std::size_t count = 0; count < numChunks; count += 1) {
               ::memcpy( ((char *) devDestAddr) + count * ld, ((char *) devOrigAddr) + count * ld, len );
            }
            return true;
         }

         virtual void _getFreeMemoryChunksList( SeparateMemoryAddressSpace const &mem, SimpleAllocator::ChunkList &list ) const {
            SimpleAllocator *sallocator = (SimpleAllocator *) mem.getSpecificData();
            sallocator->getFreeChunksList( list );
         }

   };
}

#endif
