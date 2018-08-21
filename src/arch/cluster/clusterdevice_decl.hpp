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

#ifndef _CLUSTERDEVICE_DECL
#define _CLUSTERDEVICE_DECL

#include "functor_decl.hpp"
#include "workdescriptor_decl.hpp"
#include "packer_decl.hpp"

namespace nanos {
namespace ext {

/* \brief Device specialization for cluster architecture
 * provides functions to allocate and copy data in the device
 */
   class ClusterDevice : public Device
   {
         Packer _packer;
      public:


         /*! \brief ClusterDevice constructor
          */
         ClusterDevice ( const char *n );

         /*! \brief ClusterDevice copy constructor
          */
         ClusterDevice ( const ClusterDevice &arch );

         /*! \brief ClusterDevice destructor
          */
         ~ClusterDevice();

         virtual void *memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem, WorkDescriptor const *wd, unsigned int copyIdx );
         virtual void memFree( uint64_t addr, SeparateMemoryAddressSpace &mem);
         virtual std::size_t getMemCapacity( SeparateMemoryAddressSpace &mem );
         virtual void _copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
         virtual void _copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
         virtual bool _copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
         virtual void _copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
         virtual void _copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
         virtual bool _copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
         virtual void _canAllocate( SeparateMemoryAddressSpace &mem, std::size_t *sizes, unsigned int numChunks, std::size_t *remainingSizes );
         virtual void _getFreeMemoryChunksList( SeparateMemoryAddressSpace &mem, SimpleAllocator::ChunkList &list );
   };

   extern ClusterDevice Cluster;
} // namespace ext
} // namespace nanos

#endif /* _CLUSTERDEVICE_DECL */
