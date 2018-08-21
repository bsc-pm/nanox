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

#ifndef _GPU_EVENT
#define _GPU_EVENT

#include "gpuevent_decl.hpp"
#include "debug.hpp"
#include "gpuutils.hpp"


namespace nanos {

inline void GPUEvent::updateState()
{
   if ( _timesToQuery != 0 ) {
      _timesToQuery--;
      return;
   }

   _timesToQuery = 1;

   // Check for the state of the event, to see if it has changed
   //NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_EVENT_QUERY_EVENT );
   cudaError_t recorded = cudaEventQuery( _cudaEvent );
   //NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;


   if ( recorded == cudaErrorNotReady ) {
      // This means that the event is still pending
      _state = PENDING;
#ifdef NANOS_GENERICEVENT_DEBUG
      debug( "[GPUEvt] Updating state event " << this
            << " with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
            << " and stream " << (void *) _cudaStream << " state is now PENDING"
            << ". Description: " << getDescription()
      );
#endif

      return;
   }

   // If err has any other value, something went wrong
   //fatal_cond( recorded != cudaSuccess, "Error querying for the state of a CUDA event: " +  cudaGetErrorString( err ) );

   // Since fatal_cond is only enabled in debug mode, check the returning value again
   if ( recorded == cudaSuccess ) {
      // This means that the event has been raised
      _state = RAISED;
#ifdef NANOS_GENERICEVENT_DEBUG
      debug( "[GPUEvt] Updating state event " << this
            << " with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
            << " and stream " << (void *) _cudaStream << " state is now RAISED"
            << ". Description: " << getDescription()
      );
#endif
      return;
   }
#ifdef NANOS_GENERICEVENT_DEBUG
   debug( "[GPUEvt] Updating state event " << this
         << " with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
         << " and stream " << (void *) _cudaStream
         << " state is " << stateToString() << " but I should NEVER reach this point!"
         << ". Description: " << getDescription()
   );
#endif
   // This point should not be reached: an event should always be either pending or raised

   //fatal( "CUDA error detected while updating the state of an event: " + cudaGetErrorString( recorded ) );
   fatal( cudaGetErrorString( recorded ) );

}


#ifdef NANOS_GENERICEVENT_DEBUG
inline GPUEvent::GPUEvent ( WD *wd, cudaStream_t stream, std::string desc ) : GenericEvent( wd, desc ), _timesToQuery( 1 ), _cudaStream( stream )
#else
inline GPUEvent::GPUEvent ( WD *wd, cudaStream_t stream ) : GenericEvent( wd ), _timesToQuery( 1 ), _cudaStream( stream )
#endif
{
#ifdef NANOS_GENERICEVENT_DEBUG
   debug( "[GPUEvt] Creating event " << this
         << " with wd = " << wd << " : " << ( ( wd != NULL ) ? wd->getId() : 0 )
         << " and stream " << (void *) stream
         << ". Description: " << desc
   );
#endif

   //NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_EVENT_CREATE_EVENT );
   //cudaError_t err =
   cudaEventCreateWithFlags( &_cudaEvent, cudaEventDisableTiming );
   //NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   //fatal_cond( err != cudaSuccess, "Error creating a CUDA event: " +  cudaGetErrorString( err ) );

   _state = CREATED;
}


#ifdef NANOS_GENERICEVENT_DEBUG
inline GPUEvent::GPUEvent ( WD *wd, ActionList next, cudaStream_t stream, std::string desc ) : GenericEvent( wd, next, desc )
#else
inline GPUEvent::GPUEvent ( WD *wd, ActionList next, cudaStream_t stream ) : GenericEvent( wd, next )
#endif
{
#ifdef NANOS_GENERICEVENT_DEBUG
   debug( "[GPUEvt] Creating event " << this
         << " with wd = " << wd << " : " << ( ( wd != NULL ) ? wd->getId() : 0 )
         << " and stream " << (void *) stream
         << " and " << next.size() << " elems in the queue "
         << ". Description: " << desc
   );
#endif

   //NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_EVENT_CREATE_EVENT );
   //cudaError_t err =
   cudaEventCreateWithFlags( &_cudaEvent, cudaEventDisableTiming );
   //NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   //fatal_cond( err != cudaSuccess, "Error creating a CUDA event: " +  cudaGetErrorString( err ) );

   _state = CREATED;
}


inline GPUEvent::~GPUEvent()
{
#ifdef NANOS_GENERICEVENT_DEBUG
   debug( "[GPUEvt] Destroying event " << this
         << " with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
         << " and stream " << (void *) _cudaStream
         << ". Description: " << getDescription()
   );
#endif

   ensure ( _state == RAISED, "Error trying to destroy a non-raised event" );

   //NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_EVENT_DESTROY_EVENT );
   //cudaError_t err =
   cudaEventDestroy( _cudaEvent );
   //NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   //fatal_cond( err != cudaSuccess, "Error destroying a CUDA event: " +  cudaGetErrorString( err ) );
}


inline bool GPUEvent::isPending()
{
#ifdef NANOS_GENERICEVENT_DEBUG
   debug( "[GPUEvt] Checking event " << this
         << " if pending with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
         << " and stream " << (void *) _cudaStream
         << " state is " << stateToString()
         << ". Description: " << getDescription()
   );
#endif

   // If the event is not pending, return false
   if ( _state != PENDING ) return false;

   // Otherwise, check again for the state of the event, just in case it has changed
   updateState();

   return _state == PENDING;
}


inline void GPUEvent::setPending()
{
#ifdef NANOS_GENERICEVENT_DEBUG
   debug( "[GPUEvt] Setting event " << this
         << " to pending with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
         << " and stream " << (void *) _cudaStream << " previous state was " << stateToString()
         << ". Description: " << getDescription()
   );
#endif

   //NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_EVENT_RECORD_EVENT );
   //cudaError_t err =
   cudaEventRecord( _cudaEvent, _cudaStream );
   //NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   //fatal_cond( err != cudaSuccess, "Error recording a CUDA event: " +  cudaGetErrorString( err ) );

   _state = PENDING;
}


inline bool GPUEvent::isRaised()
{
#ifdef NANOS_GENERICEVENT_DEBUG
   debug( "[GPUEvt] Checking event " << this
         << " if raised with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
         << " and stream " << (void *) _cudaStream
         << " state is " << stateToString()
         << ". Description: " << getDescription()
   );
#endif

   if ( _state == RAISED ) return true;
   if ( _state == COMPLETED ) return false;

   // Be consistent: an event should be pending before it is raised. Any other state
   // should not lead to RAISED directly
   // In the CUDA case, if an event is CREATED (this means cudaEventRecord() has not been
   // called yet) and we try to update its state, CUDA will return cudaSuccess and we
   // will think that the event has been RAISED when it has not even been recorded
   if ( _state != PENDING ) return false;

   // Otherwise, check again for the state of the event, just in case it has changed
   updateState();

#ifdef NANOS_GENERICEVENT_DEBUG
   debug( "[GPUEvt] Checking event " << this
         << " if raised after updating state with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
         << " and stream " << (void *) _cudaStream
         << " state is " << stateToString()
         << ". Description: " << getDescription()
   );
#endif

   return _state == RAISED;
}


inline void GPUEvent::setRaised()
{
#ifdef NANOS_GENERICEVENT_DEBUG
   debug( "[GPUEvt] Setting event " << this
         << " to raised with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
         << " and stream " << (void *) _cudaStream
         << " previous state was " << stateToString()
         << ". Description: " << getDescription()
   );
#endif

   //fatal_cond( !isRaised(), "Error trying to set a CUDA event to RAISED: this operation is not allowed for CUDA events" );
}


inline void GPUEvent::waitForEvent()
{
#ifdef NANOS_GENERICEVENT_DEBUG
   debug( "[GPUEvt] Waiting for event " << this
         << " with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
         << " and stream " << (void *) _cudaStream
         << " state is " << stateToString()
         << ". Description: " << getDescription()
   );
#endif

   // Event's state must be pending or raised, otherwise it is an error
   ensure ( _state != CREATED, "Error trying to wait for a non-recorded event" );

   if ( _state == RAISED ) return;

   // Check again for the state of the event, just in case it has changed
   // Force checking
   _timesToQuery = 0;
   updateState();
   if ( _state == RAISED ) return;

   //NANOS_GPU_CREATE_IN_CUDA_RUNTIME_EVENT( ext::GPUUtils::NANOS_GPU_CUDA_EVENT_SYNC_EVENT );
   //cudaError_t err =
   cudaEventSynchronize( _cudaEvent );
   //NANOS_GPU_CLOSE_IN_CUDA_RUNTIME_EVENT;

   //fatal_cond( err != cudaSuccess, "Error synchronizing with a CUDA event: " +  cudaGetErrorString( err ) );

   _state = RAISED;
}

} // namespace nanos

#endif //_GPU_EVENT
