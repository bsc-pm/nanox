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

#ifndef _FPGA_DEVICE_DECL
#define _FPGA_DEVICE_DECL

#include "workdescriptor_decl.hpp"
#include "processingelement_fwd.hpp"
#include "copydescriptor_decl.hpp"
#include "basethread.hpp" //for getMyThreadSafe() in warning/verbose, etc.

namespace nanos {

   /* \brief Device specialization for FPGA architecture
    * provides functions to allocate and copy data in the device
    */
   class FPGADevice : public Device
   {
      public:

         FPGADevice ( const char *n );

         virtual void *memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem,
                 WD const *wd, unsigned int copyIdx);
         virtual void memFree( uint64_t addr, SeparateMemoryAddressSpace &mem );

         virtual void _canAllocate( SeparateMemoryAddressSpace &mem, std::size_t *sizes,
                 unsigned int numChunks, std::size_t *remainingSizes ) {}

         virtual std::size_t getMemCapacity( SeparateMemoryAddressSpace &mem ) {
            //return 1GB of memory available. There is no addressable memory on the device
            //Just return 1GB as this is the total system memmory.
            return 1024*1024*1024;
         }
         virtual void _copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len,
               SeparateMemoryAddressSpace &mem, DeviceOps *ops,
               WD const *wd, void *hostObject, reg_t hostRegionId );

         /*! \brief Copy from remoteSrc in the host to localDst in the device
          *        Returns true if the operation is synchronous
          */
         static bool copyIn( void *localDst, CopyDescriptor &remoteSrc, size_t size, ProcessingElement *pe );

         virtual void _copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len,
               SeparateMemoryAddressSpace &mem, DeviceOps *ops,
               WorkDescriptor const *wd, void *hostObject, reg_t hostRegionId );
         /*! \brief Copy from localSrc in the device to remoteDst in the host
          *        Returns true if the operation is synchronous
          */
         static bool copyOut( CopyDescriptor &remoteDst, void *localSrc, size_t size, ProcessingElement *pe );

         /*!
          * Copy memory inside the same device. This is empty as currently does
          * not make sense for FPGA (no local memory in the fpga is accessible)
          */
         static void copyLocal( void *dst, void *src, size_t size, ProcessingElement *pe ){}

         /*!
          * Reallocate memory in the device.
          * Empty as there is no allocatable memory inside an fpga.
          */
         static void * realloc( void * address, size_t size, size_t ceSize, ProcessingElement *pe )
         {
            return NULL;
         }

         virtual bool _copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len,
               SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memorig,
               DeviceOps *ops, WorkDescriptor const *wd, void *hostObject,
               reg_t hostRegionId ) { std::cerr << "wrong copyDevToDev" <<std::endl; return false; }

         virtual void _getFreeMemoryChunksList( SeparateMemoryAddressSpace &mem,
               SimpleAllocator::ChunkList &list ) { std::cerr << "wrong _getFreeMemoryChunksList()" <<std::endl; }

         //not supported
         virtual void _copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len,
               std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace &mem,
               DeviceOps *ops, WorkDescriptor const *wd, void *hostObject,
               reg_t hostRegionId )
         {
            warning( "Strided fpga copies not implemented" );
         }

         //not supported
         virtual void _copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len,
               std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace &mem,
               DeviceOps *ops, WD const *wd, void *hostObject,
               reg_t hostRegionId )
         {
            warning( "Strided fpga copies not implemented" );
         }

         //not supported
         virtual bool _copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr,
               std::size_t len, std::size_t numChunks, std::size_t ld,
               SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig,
               DeviceOps *ops, WorkDescriptor const *wd, void *hostObject,
               reg_t hostRegionId )
         {

            warning( "Strided fpga copies not implemented" );
            return true;

         }

         /*! \brief Copy data between 2 accelerators
          * \warning addrDst and addrSrc are null for FPGA devices as there is no separate address space at this point
          * \param addrDst Copy destination address.
          * \param addrSrc Copy source address.
          * \param dstCd   Transfer copy descriptor, which contains the tag (aka. host address).
          * \param size    Data size to be copied.
          * \param peDst   Destination device.
          * \param peSrc   Source device.
          * \return Returns true if the operation is synchronous, false otherwise
          */
         static bool copyDevToDev( void * addrDst, CopyDescriptor& dstCd, void * addrSrc, std::size_t size, ProcessingElement *peDst, ProcessingElement *peSrc );

         /*!
          * \brief Finish pending transfer
          * Usually this causes to priorize a data transfer because someone else needs it
          * In this case it forces the transfer to be finished and synchronized.
          * Since all transfers are submitted, this is the way to make the data available.
          */
         static void syncTransfer( uint64_t hostAddress, ProcessingElement *pe);

   };
} // namespace nanos
#endif
