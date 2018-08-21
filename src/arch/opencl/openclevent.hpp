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

#ifndef _OpenCL_EVENT
#define _OpenCL_EVENT

#include "debug.hpp"
#include "openclevent_decl.hpp"
#include "asyncthread_decl.hpp"


namespace nanos {

inline void OpenCLEvent::updateState()
{
   if ( _timesToQuery != 0 ) {
      _timesToQuery--;
      return;
   }

   _timesToQuery = 1;

   // Check for the state of the event, to see if it has changed
   cl_int exitStatus;
   bool completed=true;
   for (int i=0; i< _numEvents && completed; ++i) {
        clGetEventInfo( _openclEvents[i],
                        CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof(cl_int),
                        &exitStatus,
                        NULL
                       );
        completed= completed && exitStatus==CL_COMPLETE;
   }

   
   if ( completed ) {
      _state = RAISED;
      debug( "[OpenCLEvt] Raising event " << this
         << " if pending with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
         << " state is " << stateToString() << " depending on " << _numEvents << " other events\n" 
        #ifdef NANOS_GENERICEVENT_DEBUG
                 << ". Description: " << getDescription()
        #endif
      );
   } else {
      _state = PENDING;
   }

}


#ifdef NANOS_GENERICEVENT_DEBUG
inline OpenCLEvent::OpenCLEvent ( WD *wd, cl_context& context, std::string desc ) : GenericEvent( wd, desc ), _timesToQuery( 1 ), _runningKernel( NULL ), _usedEvent(false)
#else
inline OpenCLEvent::OpenCLEvent ( WD *wd, cl_context& context ) : GenericEvent( wd ), _timesToQuery( 1 ), _runningKernel( NULL ), _usedEvent(false)
#endif
{   
   cl_int err;
   _numEvents=1;
   _openclEvents= NEW cl_event[_numEvents];
   _openclEvents[0]= clCreateUserEvent ( context, &err );
   
   //fatal_cond( err != CL_SUCCESS, "Error creating a OpenCL event: " +  cerr) );

   _state = CREATED;
}


#ifdef NANOS_GENERICEVENT_DEBUG
inline OpenCLEvent::OpenCLEvent ( WD *wd, ActionList next, cl_context& context, std::string desc ) : GenericEvent( wd, next, desc ), _runningKernel( NULL ), _usedEvent(false)
#else
inline OpenCLEvent::OpenCLEvent ( WD *wd, ActionList next, cl_context& context) : GenericEvent( wd, next ), _runningKernel( NULL ), _usedEvent(false)
#endif
{
   debug( "[OpenCLEvt] Creating event " << this
         << " with wd = " << wd << " : " << ( ( wd != NULL ) ? wd->getId() : 0 )
         << " and " << next.size() << " elems in the queue "
#ifdef NANOS_GENERICEVENT_DEBUG
         << ". Description: " << desc
#endif
   );

   _numEvents=1;
   _openclEvents= NEW cl_event[_numEvents];
   cl_int err;
   _openclEvents[0]= clCreateUserEvent ( context, &err );

   //fatal_cond( err != CL_SUCCESS, "Error creating a OpenCL event: " +  cerr) );

   _state = CREATED;
}


inline OpenCLEvent::~OpenCLEvent()
{
   debug( "[OpenCLEvt] Destroying event " << this
         << " with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
#ifdef NANOS_GENERICEVENT_DEBUG
         << ". Description: " << getDescription()
#endif
   );

   ensure ( _state == RAISED, "Error trying to destroy a non-raised event" );

   cl_int err= CL_SUCCESS;
   //If we depend from other events, they'll release the event
   for (int i=0 ; i< _numEvents ; ++i) {
     clReleaseEvent ( _openclEvents[i] );
   }       
   delete[] _openclEvents;
   
   //If this event was associated to a kernel, destroy the kernel
   if (_runningKernel!=NULL) {
      cl_kernel openclKernel=(cl_kernel) _runningKernel;
      clReleaseKernel( openclKernel );
   }

   fatal_cond( err != CL_SUCCESS, "Error destroying a OpenCL event: " +  err);
}


inline bool OpenCLEvent::isPending()
{
   debug( "[OpenCLEvt] Checking event " << this
         << " if pending with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
         << " state is " << stateToString()
#ifdef NANOS_GENERICEVENT_DEBUG
         << ". Description: " << getDescription()
#endif
   );

   // If the event is not pending, return false
   if ( _state != PENDING ) return false;

   // Otherwise, check again for the state of the event, just in case it has changed
   updateState();

   return _state == PENDING;
}


inline void OpenCLEvent::setPending()
{
   debug( "[OpenCLEvt] Setting event " << this
         << " to pending with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
         << " previous state was " << stateToString()
#ifdef NANOS_GENERICEVENT_DEBUG
         << ". Description: " << getDescription()
#endif
   );

   
   //If no one tried to use the event we had before this point
   //instead of using the event, wait for previous ones
   if ( !_usedEvent ) {
       clReleaseEvent ( _openclEvents[0] );   
       delete[] _openclEvents;
       nanos::ext::OpenCLThread * thread = ( nanos::ext::OpenCLThread * ) myThread;
       const AsyncThread::GenericEventList& pendingEvents=thread->getEvents();
       _numEvents=pendingEvents.size();
       _openclEvents= NEW cl_event[_numEvents];
       unsigned int localCounter=0;
       for (unsigned int i=0; i< pendingEvents.size(); ++i) {
           OpenCLEvent * evt = dynamic_cast < OpenCLEvent* > ( pendingEvents.at(i) );   
           if ( evt!=NULL && evt->_usedEvent && evt->getWD() == getWD() ) {
             _openclEvents[localCounter]=evt->getCLEvent();
             clRetainEvent(_openclEvents[localCounter]);
             localCounter++;
           } else {
             _numEvents--;
           }
       }       
   }


   //fatal_cond( err != cudaSuccess, "Error recording a CUDA event: " +  cudaGetErrorString( err ) );

   _state = PENDING;
}


inline bool OpenCLEvent::isRaised()
{
//   debug( "[OpenCLEvt] Checking event " << this
//         << " if raised with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
//         << " state is " << stateToString()
//#ifdef NANOS_GENERICEVENT_DEBUG
//         << ". Description: " << getDescription()
//#endif
//   );

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

//   debug( "[OpenCLEvt] Checking event " << this
//         << " if raised after updating state with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
//         << " state is " << stateToString()
//#ifdef NANOS_GENERICEVENT_DEBUG
//         << ". Description: " << getDescription()
//#endif
//   );

   return _state == RAISED;
}

inline cl_event& OpenCLEvent::getCLEvent() {
    _usedEvent=true;
    return _openclEvents[0];
}

inline void* OpenCLEvent::getCLKernel() {
    return _runningKernel;
}

inline void OpenCLEvent::setCLKernel(void* currKernel) {
    _runningKernel=currKernel;
}


inline void OpenCLEvent::setRaised()
{
   debug( "[OpenCLEvt] Setting event " << this
         << " to raised with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
         << " previous state was " << stateToString()
#ifdef NANOS_GENERICEVENT_DEBUG
         << ". Description: " << getDescription()
#endif
   );

   //fatal_cond( !isRaised(), "Error trying to set a CUDA event to RAISED: this operation is not allowed for CUDA events" );
}


inline void OpenCLEvent::waitForEvent()
{
   debug( "[OpenCLEvt] Waiting for event " << this
         << " with wd = " << getWD() << " : " << ( ( getWD() != NULL ) ? getWD()->getId() : 0 )
         << " state is " << stateToString()
#ifdef NANOS_GENERICEVENT_DEBUG
         << ". Description: " << getDescription()
#endif
   );

   // Event's state must be pending or raised, otherwise it is an error
   ensure ( _state != CREATED, "Error trying to wait for a non-recorded event" );

   if ( _state == RAISED ) return;

   // Check again for the state of the event, just in case it has changed
   // Force checking
   _timesToQuery = 0;
   updateState();
   if ( _state == RAISED ) return;

   cl_int err=clWaitForEvents ( _numEvents, _openclEvents );

   fatal_cond( err != CL_SUCCESS, "Error waiting for OpenCL async copy event: " + err );

   _state = RAISED;
}

} // namespace nanos

#endif //_OpenCL_EVENT
