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


namespace nanos {
namespace ext
{
   class GPUMemoryTransfer
   {
      public:
         void *                        _dst;
         void *                        _src;
         size_t                        _size;
         bool                          _requested;

         GPUMemoryTransfer( void * dest, void * source, size_t s ) :
            _dst( dest ), _src( source ), _size( s ) {}

         ~GPUMemoryTransfer() {}

         GPUMemoryTransfer& operator=( const nanos::ext::GPUMemoryTransfer &mt )
         {
            if ( &mt == this ) return *this;

            _dst = mt._dst;
            _src = mt._src;
            _size = mt._size;
            _requested = mt._requested;
            return *this;
         }
   };

   class GPUMemoryTransferList
   {
      public:

         GPUMemoryTransferList() {}
         virtual ~GPUMemoryTransferList() {}

         virtual void addMemoryTransfer ( void * dest, void * source, size_t size ) {}
         virtual void addMemoryTransfer ( uint64_t address ) {}
         virtual void removeMemoryTransfer ( void * dstAddress ) {}
         virtual void removeMemoryTransfer () {}
         virtual void checkAddressForMemoryTransfer ( void * address ) {}
         virtual void executeMemoryTransfers () {}
         virtual void requestTransfer( void *  address ) {}
         virtual void clearMemoryTransfers () {}
         virtual void reset () {}
   };

   class GPUMemoryTransferOutSyncList : public GPUMemoryTransferList
   {
      public:

         GPUMemoryTransferOutSyncList() : GPUMemoryTransferList() {}
         ~GPUMemoryTransferOutSyncList() {}

         void addMemoryTransfer ( void * dest, void * source, size_t size );
   };

   class GPUMemoryTransferOutAsyncList : public GPUMemoryTransferList
   {
      private:
         std::list<GPUMemoryTransfer>   _pendingTransfersAsync;

      public:
         GPUMemoryTransferOutAsyncList() : GPUMemoryTransferList() {}
         ~GPUMemoryTransferOutAsyncList()
         {
            if ( !_pendingTransfersAsync.empty() ) {
               warning ( "Attempting to delete the output pending transfers list with already "
                     << _pendingTransfersAsync.size() << " pending transfers to perform" );
            }
         }

         void addMemoryTransfer ( void * dest, void * source, size_t size )
         {
            _pendingTransfersAsync.push_back( GPUMemoryTransfer ( dest, source, size ) );
         }

         void removeMemoryTransfer ( std::list<GPUMemoryTransfer>::iterator it );

         void removeMemoryTransfer ( void * dstAddress );

         void removeMemoryTransfer ()
         {
            if ( !_pendingTransfersAsync.empty() ) {
               bool found = false;

               for ( std::list<GPUMemoryTransfer>::iterator it = _pendingTransfersAsync.begin();
                     it != _pendingTransfersAsync.end(); it++ ) {
                  if ( it->_requested ) {
                     found = true;
                     removeMemoryTransfer( it );
                     break;
                  }
               }

               if ( !found ) {
                  removeMemoryTransfer( _pendingTransfersAsync.begin() );
               }
            }
         }

         void checkAddressForMemoryTransfer ( void * address )
         {
            for ( std::list<GPUMemoryTransfer>::iterator it = _pendingTransfersAsync.begin();
                  it != _pendingTransfersAsync.end();
                  it++ ) {
               if ( it->_src == address ) {
                  removeMemoryTransfer( it );
               }
            }
         }

         void executeMemoryTransfers ();

         void requestTransfer( void *  address )
         {
            for ( std::list<GPUMemoryTransfer>::iterator it = _pendingTransfersAsync.begin();
                  it != _pendingTransfersAsync.end(); it++ ) {
               if ( it->_dst == address ) {
                  it->_requested = true;
               }
            }
         }

         void finishMemoryTransfer ( std::list<GPUMemoryTransfer>::iterator it );

         std::list<GPUMemoryTransfer>& getList()
         {
          return _pendingTransfersAsync;
         }
   };

   class GPUMemoryTransferInAsyncList : public GPUMemoryTransferList
   {
      private:
         std::list<uint64_t>   _pendingTransfersAsync;

      public:
         GPUMemoryTransferInAsyncList() : GPUMemoryTransferList() {}
         ~GPUMemoryTransferInAsyncList()
         {
            if ( !_pendingTransfersAsync.empty() ) {
               warning ( "Attempting to delete the input pending transfers list with already "
                     << _pendingTransfersAsync.size() << " pending transfers to perform" );
            }
         }

         void addMemoryTransfer ( uint64_t address )
         {
            _pendingTransfersAsync.push_back( address );
         }

         void reset ()
         {
            _pendingTransfersAsync.clear();
         }

         void clearMemoryTransfers ();
   };

}
}

#endif

