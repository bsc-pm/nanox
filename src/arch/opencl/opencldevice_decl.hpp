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

#ifndef _OpenCL_DEVICE_DECL
#define _OpenCL_DEVICE_DECL

#include "workdescriptor.hpp"
#include "copydescriptor_decl.hpp"
#include "processingelement_fwd.hpp"

namespace nanos {

class OpenCLDevice : public Device
{
public:
   OpenCLDevice ( const char *name );

public:
//   static void *allocate( size_t size, ProcessingElement *pe, uint64_t tag = NULL  );
//
//   static void *realloc( void * address,
//                         size_t size,
//                         size_t ceSize,
//                         ProcessingElement *pe );
//
//   static void free( void *address, ProcessingElement *pe );
//
//   static bool copyDevToDev( void *addrDst,
//                             CopyDescriptor& dstCd,
//                             void* addrSrc,
//                             size_t size,
//                             ProcessingElement *peDst,
//                             ProcessingElement *peSrc );
//   
//   static bool copyIn( void *localDst,
//                       CopyDescriptor &remoteSrc,
//                       size_t size,
//                       ProcessingElement *pe );
//
//   static bool copyOut( CopyDescriptor &remoteDst,
//                        void *localSrc,
//                        size_t size,
//                        ProcessingElement *pe );
//
//   static void copyLocal( void *dst,
//                          void *src,
//                          size_t size,
//                          ProcessingElement *pe )
//   {
//       return;
//   }
//
//   static void syncTransfer( uint64_t hostAddress, ProcessingElement *pe );
//   
   virtual void *memAllocate( std::size_t size, SeparateMemoryAddressSpace &mem, WD const *wd, unsigned int copyIdx );
   virtual void memFree( uint64_t addr, SeparateMemoryAddressSpace &mem );
   virtual void _canAllocate( SeparateMemoryAddressSpace &mem, std::size_t *sizes, unsigned int numChunks, std::size_t *remainingSizes );
   virtual std::size_t getMemCapacity( SeparateMemoryAddressSpace &mem );

   virtual void _copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
   virtual void _copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
   virtual bool _copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
   virtual void _copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
   virtual void _copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t count, std::size_t ld, SeparateMemoryAddressSpace &mem, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
   virtual bool _copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, std::size_t numChunks, std::size_t ld, SeparateMemoryAddressSpace &memDest, SeparateMemoryAddressSpace &memOrig, DeviceOps *ops, WD const *wd, void *hostObject, reg_t hostRegionId );
   virtual void _getFreeMemoryChunksList( SeparateMemoryAddressSpace &mem, SimpleAllocator::ChunkList &list );

};

} // namespace nanos

#endif // _OpenCL_DEVICE_DECL
