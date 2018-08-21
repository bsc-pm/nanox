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

#ifndef _NANOS_MEMORY_TRANSFER
#define _NANOS_MEMORY_TRANSFER

#include "gpumemorytransfer_decl.hpp"
#include "basethread.hpp"


namespace nanos {
namespace ext {

void GPUMemoryTransfer::completeTransfer()
{
   _hostAddress._ops->completeOp();
   delete this;
}

 
GPUMemoryTransferOutList::~GPUMemoryTransferOutList()
{
   if ( !_pendingTransfersAsync.empty() ) {
      warning ( "Attempting to delete the output pending transfers list with already "
            << _pendingTransfersAsync.size() << " pending transfers to perform" );
   }
}

void GPUMemoryTransferOutList::addMemoryTransfer ( CopyDescriptor &hostAddress, void * deviceAddress, size_t len, size_t count, size_t ld )
{
   GPUMemoryTransfer * mt = NEW GPUMemoryTransfer ( hostAddress, deviceAddress, len, count, ld );
   _lock.acquire();
   _pendingTransfersAsync.push_back( mt );
   _lock.release();
}


void GPUMemoryTransferOutAsyncList::addMemoryTransfer ( CopyDescriptor &hostAddress, void * deviceAddress, size_t len, size_t count, size_t ld )
{
   GPUMemoryTransfer * mt = NEW GPUMemoryTransfer ( hostAddress, deviceAddress, len, count, ld );
   _lock.acquire();
   _pendingTransfersAsync.push_back( mt );
   _lock.release();
}

void GPUMemoryTransferOutAsyncList::executeMemoryTransfers ()
{
   executeMemoryTransfers( _pendingTransfersAsync );
}


GPUMemoryTransferInAsyncList::~GPUMemoryTransferInAsyncList()
{
   ensure( _pendingTransfersAsync.empty(),
         "Attempting to delete the input pending transfers list with already "
         + toString<size_t>( _pendingTransfersAsync.size() ) + " pending transfers to perform" );

   ensure( _requestedTransfers.empty(),
                     "Attempting to delete the requested input transfers list with already "
                     + toString<size_t>( _requestedTransfers.size() ) + " pending transfers to perform" );
}

void GPUMemoryTransferInAsyncList::addMemoryTransfer ( CopyDescriptor &hostAddress, void * deviceAddress, size_t len, size_t count, size_t ld )
{
   GPUMemoryTransfer * mt = NEW GPUMemoryTransfer ( hostAddress, deviceAddress, len, count, ld );
   _lock.acquire();
   _requestedTransfers.push_back( mt );
   _lock.release();
}

} // namespace ext
} // namespace nanos

#endif // _NANOS_MEMORY_TRANSFER
