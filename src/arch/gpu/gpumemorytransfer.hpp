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

#ifndef _NANOS_MEMORY_TRANSFER
#define _NANOS_MEMORY_TRANSFER

#include "gpumemorytransfer_decl.hpp"
#include "basethread_fwd.hpp"
#include "debug.hpp"

using namespace nanos;
using namespace nanos::ext;


void nanos::ext::GPUMemoryTransferOutList::addMemoryTransfer ( CopyDescriptor &hostAddress, void * deviceAddress, size_t size )
{
   _lock.acquire();
   _pendingTransfersAsync.push_back( *NEW GPUMemoryTransfer ( hostAddress, deviceAddress, size ) );
   _lock.release();
}


void nanos::ext::GPUMemoryTransferOutAsyncList::executeMemoryTransfers ()
{
   executeMemoryTransfers( _pendingTransfersAsync );
}


void nanos::ext::GPUMemoryTransferInAsyncList::addMemoryTransfer ( CopyDescriptor &address )
{
   _pendingTransfersAsync.push_back( address );
}

void nanos::ext::GPUMemoryTransferInAsyncList::addMemoryTransfer ( CopyDescriptor &hostAddress, void * deviceAddress, size_t size )
{
   _lock.acquire();
   _requestedTransfers.push_back( *NEW GPUMemoryTransfer ( hostAddress, deviceAddress, size ) );
   _lock.release();
}


#endif // _NANOS_MEMORY_TRANSFER
