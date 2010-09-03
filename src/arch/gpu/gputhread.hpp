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

#ifndef _NANOS_GPU_THREAD
#define _NANOS_GPU_THREAD

#include "compatibility.hpp"
#include "gpudd.hpp"
#include "smpthread.hpp"


namespace nanos {
namespace ext
{

   class GPUThread : public SMPThread
   {

      friend class GPUProcessor;

      class PendingCopiesList
      {
         public:

            PendingCopiesList() {}
            ~PendingCopiesList() {}

            virtual void addPendingCopy ( void * dest, void * source, size_t size ) {}
            virtual void removePendingCopy () {}
            virtual void checkAddressForPendingCopy ( void * address ) {}
            virtual void executePendingCopies () {}
      };

      class PendingCopiesSyncList : public PendingCopiesList
      {
         public:

            PendingCopiesSyncList() : PendingCopiesList() {}
            ~PendingCopiesSyncList() {}

            void addPendingCopy ( void * dest, void * source, size_t size )
            {
               GPUDevice::copyOutSyncToHost( dest, source, size );
            }
      };


      class PendingCopiesAsyncList : public PendingCopiesList
      {
         private:

            class PendingCopy : public nanos::WG
            {
               public:
                  void *                        _dst;
                  void *                        _src;
                  size_t                        _size;
                  TR1::shared_ptr<DOSubmit>     _do;

                  PendingCopy( void * dest, void * source, size_t s ) : _dst( dest ), _src( source ),
                        _size( s ), _do( myThread->getCurrentWD()->getDOSubmit() )
                  {
                     if ( _do != NULL ) {
                        _do->increaseReferences();
                     }
                  }

                  ~PendingCopy() {}
            };


            std::vector<PendingCopy>   _pendingCopiesAsync;

         public:
            PendingCopiesAsyncList() : PendingCopiesList() {}
            ~PendingCopiesAsyncList()
            {
               if ( !_pendingCopiesAsync.empty() ) {
                  warning ( "Attempting to delete the pending copies list with already "
                        << _pendingCopiesAsync.size()
                        << " pending copies to perform" );
               }
            }

            void addPendingCopy ( void * dest, void * source, size_t size )
            {
               _pendingCopiesAsync.push_back( PendingCopy( dest, source, size ) );
               myThread->getCurrentWD()->getParent()->addWork( _pendingCopiesAsync.back() );
            }

            void removePendingCopy ( std::vector<PendingCopy>::iterator it );

            void removePendingCopy ()
            {
               if ( !_pendingCopiesAsync.empty() ) {
                  removePendingCopy( _pendingCopiesAsync.begin() );
               }
            }

            void checkAddressForPendingCopy ( void * address )
            {
               for ( std::vector<PendingCopy>::iterator it = _pendingCopiesAsync.begin();
                     it != _pendingCopiesAsync.end();
                     it++ ) {
                  if ( it->_src == address ) {
                     removePendingCopy( it );
                  }
               }
            }

            void executePendingCopies ();

            void finishPendingCopy ( std::vector<PendingCopy>::iterator it );
      };

      private:
         int                        _gpuDevice; // Assigned GPU device Id
         PendingCopiesList *        _pendingCopies;

         // disable copy constructor and assignment operator
         GPUThread( const GPUThread &th );
         const GPUThread & operator= ( const GPUThread &th );

      public:
         // constructor
         GPUThread( WD &w, PE *pe, int device ) : SMPThread( w, pe ), _gpuDevice( device ) {}

         // destructor
         virtual ~GPUThread() {}

         virtual void runDependent ( void );

         virtual void inlineWorkDependent( WD &work );

         virtual void yield();

         void idle()
         {
            _pendingCopies->removePendingCopy();
         }

         int getGpuDevice ()
         {
            return _gpuDevice;
         }

         void addPendingCopy ( void * dest, void * source, size_t size )
         {
            _pendingCopies->addPendingCopy( dest, source, size );
         }

         void checkAddressForPendingCopy ( void * address )
         {
            _pendingCopies->checkAddressForPendingCopy( address );
         }

         void executePendingCopies ()
         {
            _pendingCopies->executePendingCopies();
         }
   };


}
}

#endif
