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

        /* \breif allocate size bytes in the device
         */
         static void * allocate( size_t size, ProcessingElement *pe, uint64_t tag = 0 )
         {
#ifdef CLUSTER_DEV
            char * addr = (char *)0xdeadbeef;
#else
            char *addr = NEW char[size];
#endif
            return addr; 
         }

        /* \brief free address
         */
         static void free( void *address, ProcessingElement *pe )
         {
#ifdef CLUSTER_DEV
#else
            delete[] (char *) address;
#endif
         }

        /* \brief Reallocate and copy from address.
         */
         static void * realloc( void *address, size_t size, size_t old_size, ProcessingElement *pe )
         {
            return ::realloc( address, size );
         }

        /* \brief Copy from remoteSrc in the host to localDst in the device
         *        Returns true if the operation is synchronous
         */
         static bool copyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size, ProcessingElement *pe )
         {
#ifdef CLUSTER_DEV
#else
            memcpy( localDst, (void *)remoteSrc.getTag(), size );
#endif
            return true;
         }

        /* \brief Copy from localSrc in the device to remoteDst in the host
         *        Returns true if the operation is synchronous
         */
         static bool copyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size, ProcessingElement *pe )
         {
#ifdef CLUSTER_DEV
#else
            memcpy( (void *)remoteDst.getTag(), localSrc, size );
#endif
            return true;
         }

        /* \brief Copy localy in the device from src to dst
         */
         static void copyLocal( void *dst, void *src, size_t size, ProcessingElement *pe )
         {
#ifdef CLUSTER_DEV
            memcpy( dst, src, size );
#else
            memcpy( dst, src, size );
#endif
         }

         static void syncTransfer( uint64_t hostAddress, ProcessingElement *pe)
         {
         }

         static bool copyDevToDev( void * addrDst, CopyDescriptor& cdDst, void * addrSrc, std::size_t size, ProcessingElement *peDst, ProcessingElement *peSrc )
         {
            return true;
         }

         virtual void *memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem, uint64_t targetHostAddr = 0 ) const {
            void *retAddr = NULL;

            SimpleAllocator *sallocator = (SimpleAllocator *) mem.getSpecificData();
            sallocator->lock();
            retAddr = sallocator->allocate( size );
            sallocator->unlock();
            return retAddr;
         }

         virtual void memFree( uint64_t addr, SeparateMemoryAddressSpace &mem ) const {
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

         /*
         virtual void _copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace const &mem, DeviceOps *ops, Functor *f, WorkDescriptor const &wd, void *hostObject, reg_t hostRegionId ) {
            std::cerr << "wrong copyIn" <<std::endl;
         }

         virtual void _copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace const &mem, DeviceOps *ops, Functor *f, WorkDescriptor const &wd, void *hostObject, reg_t hostRegionId ) {
            std::cerr << "wrong copyOut" <<std::endl;
         }

         virtual bool _copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace const &memDest, SeparateMemoryAddressSpace const &memOrig, DeviceOps *ops, Functor *f, WorkDescriptor const &wd, void *hostObject, reg_t hostRegionId ) const {
            std::cerr << "wrong copyDevToDev" <<std::endl; return false;
         }*/

         virtual void _getFreeMemoryChunksList( SeparateMemoryAddressSpace const &mem, SimpleAllocator::ChunkList &list ) const {
            SimpleAllocator *sallocator = (SimpleAllocator *) mem.getSpecificData();
            sallocator->getFreeChunksList( list );
         }

   };
}

#endif
