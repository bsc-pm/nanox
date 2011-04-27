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

#include "basethread.hpp"
#include "compatibility.hpp"
#include "copydescriptor_decl.hpp"


namespace nanos {
namespace ext
{
   class GPUMemoryTransfer
   {
      public:
         CopyDescriptor _hostAddress; 
         void *         _deviceAddress;
         size_t         _size;
         bool           _requested;

         GPUMemoryTransfer( CopyDescriptor &hostAddress, void * deviceAddress, size_t s ) :
            _hostAddress( hostAddress ), _deviceAddress( deviceAddress ), _size( s ), _requested( false ) {}

         GPUMemoryTransfer( GPUMemoryTransfer &mt ) :
            _hostAddress( mt._hostAddress ), _deviceAddress( mt._deviceAddress ), _size( mt._size ),
            _requested( mt._requested ) {}

         GPUMemoryTransfer ( const GPUMemoryTransfer &mt ) :
                     _hostAddress( mt._hostAddress ), _deviceAddress( mt._deviceAddress ), _size( mt._size ),
                     _requested( mt._requested ) {}

         ~GPUMemoryTransfer() {}
   };

   class GPUMemoryTransferList
   {
      public:

         GPUMemoryTransferList() {}
         virtual ~GPUMemoryTransferList() {}

         virtual void addMemoryTransfer ( CopyDescriptor &hostAddress, void * deviceAddress, size_t size ) {}
         virtual void addMemoryTransfer ( CopyDescriptor &hostAddress ) {}
         virtual void removeMemoryTransfer ( CopyDescriptor &hostAddress ) {}
         virtual void removeMemoryTransfer () {}
         virtual void checkAddressForMemoryTransfer ( void * address ) {}
         virtual void executeMemoryTransfers () {}
         virtual void requestTransfer( void * address ) {}
         virtual void clearMemoryTransfers () {}
   };

   class GPUMemoryTransferOutList : public GPUMemoryTransferList
   {
      protected:
         std::list<GPUMemoryTransfer>   _pendingTransfersAsync;
         Lock                           _lock;

      public:
         GPUMemoryTransferOutList() : GPUMemoryTransferList(), _lock() {}

         virtual ~GPUMemoryTransferOutList()
         {
            if ( !_pendingTransfersAsync.empty() ) {
               warning ( "Attempting to delete the output pending transfers list with already "
                     << _pendingTransfersAsync.size() << " pending transfers to perform" );
            }
         }

         virtual void addMemoryTransfer ( CopyDescriptor &hostAddress, void * deviceAddress, size_t size )
         {
            _lock.acquire();
            _pendingTransfersAsync.push_back( *NEW GPUMemoryTransfer ( hostAddress, deviceAddress, size ) );
            _lock.release();
         }

         virtual void removeMemoryTransfer ( GPUMemoryTransfer &mt ) {}

         virtual void removeMemoryTransfer ()
         {
            if ( !_pendingTransfersAsync.empty() ) {
               bool found = false;
               for ( std::list<GPUMemoryTransfer>::iterator it = _pendingTransfersAsync.begin();
                     it != _pendingTransfersAsync.end(); it++ ) {
                  _lock.acquire();
                  if ( it->_requested ) {
                     found = true;
                     GPUMemoryTransfer mt ( *it );
                     _pendingTransfersAsync.erase( it );
                     _lock.release();
                     removeMemoryTransfer( mt );
                     break;
                  }
                  _lock.release();
              }

               if ( !found ) {
                  _lock.acquire();
                  GPUMemoryTransfer mt ( *_pendingTransfersAsync.begin() );
                  _pendingTransfersAsync.erase( _pendingTransfersAsync.begin() );
                  _lock.release();
                  removeMemoryTransfer( mt );
               }
            }
         }

         virtual void checkAddressForMemoryTransfer ( void * address )
         {
            for ( std::list<GPUMemoryTransfer>::iterator it = _pendingTransfersAsync.begin();
                  it != _pendingTransfersAsync.end();
                  it++ ) {
               _lock.acquire();
               if ( it->_hostAddress.getTag() == ( uint64_t ) address ) {
                  GPUMemoryTransfer mt ( *it );
                  _pendingTransfersAsync.erase( it );
                  _lock.release();
                  removeMemoryTransfer( mt );
                  _lock.acquire();
               }
               _lock.release();
            }
         }

         virtual void requestTransfer( void * address )
         {
            _lock.acquire();
            for ( std::list<GPUMemoryTransfer>::iterator it = _pendingTransfersAsync.begin();
                  it != _pendingTransfersAsync.end(); it++ ) {
               if ( it->_hostAddress.getTag() == ( uint64_t ) address ) {
                  it->_requested = true;
               }
            }
            _lock.release();
         }
   };

   class GPUMemoryTransferOutSyncList : public GPUMemoryTransferOutList
   {
      public:

         GPUMemoryTransferOutSyncList() : GPUMemoryTransferOutList() {}
         ~GPUMemoryTransferOutSyncList() {}

         void removeMemoryTransfer ( GPUMemoryTransfer &mt );

         void executeMemoryTransfers ()
         {
            while ( !_pendingTransfersAsync.empty() ) {
               _lock.acquire();
               GPUMemoryTransfer mt ( *_pendingTransfersAsync.begin() );
               _pendingTransfersAsync.erase( _pendingTransfersAsync.begin() );
               _lock.release();

               removeMemoryTransfer( mt );
            }
         }
   };

   class GPUMemoryTransferOutAsyncList : public GPUMemoryTransferOutList
   {
      public:
         GPUMemoryTransferOutAsyncList() : GPUMemoryTransferOutList() {}
         ~GPUMemoryTransferOutAsyncList() {}

         void removeMemoryTransfer ( GPUMemoryTransfer &mt );

         void removeMemoryTransfer ( CopyDescriptor &hostAddress );

         void executeMemoryTransfers ();
   };

   class GPUMemoryTransferInAsyncList : public GPUMemoryTransferList
   {
      private:
         std::list<CopyDescriptor>        _pendingTransfersAsync;
         std::list<GPUMemoryTransfer>     _requestedTransfers;
         Lock                             _lock;

      public:
         GPUMemoryTransferInAsyncList() : GPUMemoryTransferList(), _lock() {}
         ~GPUMemoryTransferInAsyncList()
         {
            ensure( _pendingTransfersAsync.empty(),
                  "Attempting to delete the input pending transfers list with already "
                  + toString<size_t>( _pendingTransfersAsync.size() ) + " pending transfers to perform" );
         }

         void addMemoryTransfer ( CopyDescriptor &address )
         {
            _pendingTransfersAsync.push_back( address );
         }

         void addMemoryTransfer ( CopyDescriptor &hostAddress, void * deviceAddress, size_t size )
         {
            _lock.acquire();
            _requestedTransfers.push_back( *NEW GPUMemoryTransfer ( hostAddress, deviceAddress, size ) );
            _lock.release();
         }

         void clearMemoryTransfers ();

         void removeMemoryTransfer ( GPUMemoryTransfer &mt );

         void executeMemoryTransfers ();
   };

}
}

#endif
