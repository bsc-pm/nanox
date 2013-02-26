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

#ifndef _GPU_EVENT_DECL
#define _GPU_EVENT_DECL

#include "genericevent_decl.hpp"
#include "debug.hpp"
#include "gpuconfig.hpp"

#include <cuda_runtime.h>

//#include <queue>
//#include <functional>

//#include "workdescriptor_fwd.hpp"

namespace nanos
{

   class GPUEvent : public GenericEvent
   {
      private:
         int            _timesToQuery;
         cudaEvent_t    _cudaEvent;
         cudaStream_t   _cudaStream;

         void updateState()
         {
            if ( _timesToQuery != 0 ) {
               _timesToQuery--;
               return;
            }

            _timesToQuery = 100;

            // Check for the state of the event, to see if it has changed
            NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_EVENT_QUERY_EVENT );
            cudaError_t recorded = cudaEventQuery( _cudaEvent );
            NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;


            if ( recorded == cudaErrorNotReady ) {
               // This means that the event is still pending
               _state = PENDING;
               debug( "[GPUEvt] Updating state event " << this
                     << " with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
                     << " and stream " << (void *) _cudaStream << " state is now PENDING" );

               return;
            }

            // If err has any other value, something went wrong
            //fatal_cond( recorded != cudaSuccess, "Error querying for the state of a CUDA event: " +  cudaGetErrorString( err ) );

            // Since fatal_cond is only enabled in debug mode, check the returning value again
            if ( recorded == cudaSuccess ) {
               // This means that the event has been raised
               _state = RAISED;
               debug( "[GPUEvt] Updating state event " << this
                     << " with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
                     << " and stream " << (void *) _cudaStream << " state is now RAISED" );
               return;
            }
            debug( "[GPUEvt] Updating state event " << this
                  << " with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
                  << " and stream " << (void *) _cudaStream
                  << " state is " << _state << " but I should NEVER reach this point!" );
            // This point should not be reached: an event should always be either pending or raised

            //fatal( "CUDA error detected while updating the state of an event: " + cudaGetErrorString( recorded ) );
            fatal( cudaGetErrorString( recorded ) );

         }

      public:
        /*! \brief GPUEvent constructor
         */
         GPUEvent ( WD *wd, cudaStream_t stream = 0 ) : GenericEvent( wd ), _timesToQuery( 100 ), _cudaStream( stream )
         {
            debug( "[GPUEvt] Creating event " << this
                  << " with wd = " << wd << " : " << ( ( wd != NULL ) ? wd->getId() : 0 )
                  << " and stream " << (void *) stream );

            NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_EVENT_CREATE_EVENT );
            //cudaError_t err =
            cudaEventCreateWithFlags( &_cudaEvent, cudaEventDisableTiming );
            NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

            //fatal_cond( err != cudaSuccess, "Error creating a CUDA event: " +  cudaGetErrorString( err ) );

            _state = CREATED;
         }

         /*! \brief GPUEvent constructor
          */
         GPUEvent ( WD *wd, std::queue<Action *> next, cudaStream_t stream = 0 ) : GenericEvent( wd, next )
         {
            debug( "[GPUEvt] Creating event " << this
                  << " with wd = " << wd << " : " << ( ( wd != NULL ) ? wd->getId() : 0 )
                  << " and stream " << (void *) stream
                  << " and " << next.size() << " elems in the queue " );

            NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_EVENT_CREATE_EVENT );
            //cudaError_t err =
            cudaEventCreateWithFlags( &_cudaEvent, cudaEventDisableTiming );
            NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

            //fatal_cond( err != cudaSuccess, "Error creating a CUDA event: " +  cudaGetErrorString( err ) );

            _state = CREATED;
         }

        /*! \brief GPUEvent destructor
         */
         ~GPUEvent()
         {
            debug( "[GPUEvt] Destroying event " << this
                  << " with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
                  << " and stream " << (void *) _cudaStream );

            ensure ( _state == RAISED, "Error trying to destroy a non-raised event" );

            NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_EVENT_DESTROY_EVENT );
            //cudaError_t err =
            cudaEventDestroy( _cudaEvent );
            NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

            //fatal_cond( err != cudaSuccess, "Error destroying a CUDA event: " +  cudaGetErrorString( err ) );
         }

         // set/get methods
         bool isPending()
         {
            debug( "[GPUEvt] Checking event " << this
                  << " if pending with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
                  << " and stream " << (void *) _cudaStream
                  << " state is " << _state );

            // If the event is not pending, return false
            if ( _state != PENDING ) return false;

            // Otherwise, check again for the state of the event, just in case it has changed
            updateState();

            return _state == PENDING;
         }

         void setPending()
         {
            debug( "[GPUEvt] Setting event " << this
                  << " to pending with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
                  << " and stream " << (void *) _cudaStream << " previous state was " << _state );

            NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_EVENT_RECORD_EVENT );
            //cudaError_t err =
            cudaEventRecord( _cudaEvent, _cudaStream );
            NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

            //fatal_cond( err != cudaSuccess, "Error recording a CUDA event: " +  cudaGetErrorString( err ) );

            _state = PENDING;
         }

         bool isRaised()
         {
            debug( "[GPUEvt] Checking event " << this
                  << " if raised with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
                  << " and stream " << (void *) _cudaStream
                  << " state is " << _state );

            if ( _state == RAISED ) return true;

            // Otherwise, check again for the state of the event, just in case it has changed
            updateState();

            return _state == RAISED;
         }

         void setRaised()
         {
            debug( "[GPUEvt] Setting event " << this
                  << " to raised with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
                  << " and stream " << (void *) _cudaStream
                  << " previous state was " << _state );

            //fatal_cond( !isRaised(), "Error trying to set a CUDA event to RAISED: this operation is not allowed for CUDA events" );
         }

         // event synchronization related methods
         void waitForEvent()
         {
            debug( "[GPUEvt] Waiting for event " << this
                  << " with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
                  << " and stream " << (void *) _cudaStream
                  << " state is " << _state );

            // Event's state must be pending or raised, otherwise it is an error
            ensure ( _state != CREATED, "Error trying to wait for a non-recorded event" );

            if ( _state == RAISED ) return;

            // Check again for the state of the event, just in case it has changed
            // Force checking
            _timesToQuery = 0;
            updateState();
            if ( _state == RAISED ) return;

            NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::NANOS_GPU_CUDA_EVENT_SYNC_EVENT );
            //cudaError_t err =
            cudaEventSynchronize( _cudaEvent );
            NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

            //fatal_cond( err != cudaSuccess, "Error synchronizing with a CUDA event: " +  cudaGetErrorString( err ) );

            _state = RAISED;
         }
   };
}

#endif //_GPU_EVENT_DECL
