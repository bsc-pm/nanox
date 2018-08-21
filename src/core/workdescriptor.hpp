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

#ifndef _NANOS_WORK_DESCRIPTOR_H
#define _NANOS_WORK_DESCRIPTOR_H

#include <stdlib.h>
#include <utility>
#include <vector>
#include "workdescriptor_decl.hpp"
#include "dependableobjectwd.hpp"
#include "copydata.hpp"
#include "synchronizedcondition_decl.hpp"
#include "atomic.hpp"
#include "lazy.hpp"
#include "instrumentationcontext.hpp"
#include "schedule.hpp"
#include "dependenciesdomain.hpp"
#include "allocator_decl.hpp"
#include "system.hpp"
#include "slicer_decl.hpp"

namespace nanos {

inline WorkDescriptor::WorkDescriptor ( int ndevices, DeviceData **devs, size_t data_size, size_t data_align, void *wdata,
                                 size_t numCopies, CopyData *copies, nanos_translate_args_t translate_args, const char *description )
                               : _id( sys.getWorkDescriptorId() ), _hostId(0), _components( 0 ),
                                 _componentsSyncCond( EqualConditionChecker<int>( &_components.override(), 0 ) ), _parent(NULL), _forcedParent(NULL),
                                 _data_size ( data_size ), _data_align( data_align ),  _data ( wdata ), _totalSize(0),
                                 _wdData ( NULL ), _scheduleData( NULL ),
                                 _flags(), _tiedTo ( NULL ), _tiedToLocation( (memory_space_id_t) -1 ),
                                 _state( INIT ), _syncCond( NULL ),  _myQueue ( NULL ), _depth ( 0 ),
                                 _numDevices ( ndevices ), _devices ( devs ), _activeDeviceIdx( ndevices == 1 ? 0 : ndevices ),
#ifdef GPU_DEV
                                 _cudaStreamIdx( -1 ),
#endif
                                 _numCopies( numCopies ), _copies( copies ), _paramsSize( 0 ),
                                 _versionGroupId( 0 ), _executionTime( 0.0 ), _estimatedExecTime( 0.0 ),
                                 _doSubmit(NULL), _doWait(), _depsDomain( sys.getDependenciesManager()->createDependenciesDomain() ),
                                 _translateArgs( translate_args ),
                                 _priority( 0 ), _commutativeOwnerMap(NULL), _commutativeOwners(NULL),
                                 _copiesNotInChunk(false), _description(description), _instrumentationContextData(), _slicer(NULL),
                                 _taskReductions(),
                                 _notifyCopy( NULL ), _notifyThread( NULL ), _remoteAddr( NULL ), _callback(0), _arguments(0),
                                 _submittedWDs( NULL ), _reachedTaskwait( false ), _schedPredecessorLocs(),
                                 _mcontrol( this, numCopies )
                                 {
                                    _flags.is_final = 0;
                                    _flags.is_submitted = false;
                                    _flags.is_recoverable = false;
                                    _flags.is_invalid = false;
                                    if ( copies != NULL ) {
                                       for ( unsigned int i = 0; i < numCopies; i += 1 ) {
                                          copies[i].setHostBaseAddress( 0 );
                                          copies[i].setRemoteHost( false );
                                       }
                                    }
                                    for (unsigned int __i=0; __i<8;__i+=1) {
                                       _schedValues[__i]=-1;
                                    }
                                 }

inline WorkDescriptor::WorkDescriptor ( DeviceData *device, size_t data_size, size_t data_align, void *wdata,
                                 size_t numCopies, CopyData *copies, nanos_translate_args_t translate_args, const char *description )
                               : _id( sys.getWorkDescriptorId() ), _hostId( 0 ), _components( 0 ),
                                 _componentsSyncCond( EqualConditionChecker<int>( &_components.override(), 0 ) ), _parent(NULL), _forcedParent(NULL),
                                 _data_size ( data_size ), _data_align ( data_align ), _data ( wdata ), _totalSize(0),
                                 _wdData ( NULL ), _scheduleData( NULL ),
                                 _flags(), _tiedTo ( NULL ), _tiedToLocation( (memory_space_id_t) -1 ),
                                 _state( INIT ), _syncCond( NULL ), _myQueue ( NULL ), _depth ( 0 ),
                                 _numDevices ( 1 ), _devices ( NULL ), _activeDeviceIdx( 0 ),
#ifdef GPU_DEV
                                 _cudaStreamIdx( -1 ),
#endif
                                 _numCopies( numCopies ), _copies( copies ), _paramsSize( 0 ),
                                 _versionGroupId( 0 ), _executionTime( 0.0 ), _estimatedExecTime( 0.0 ),
                                 _doSubmit(NULL), _doWait(), _depsDomain( sys.getDependenciesManager()->createDependenciesDomain() ),
                                 _translateArgs( translate_args ),
                                 _priority( 0 ),  _commutativeOwnerMap(NULL), _commutativeOwners(NULL),
                                 _copiesNotInChunk(false), _description(description), _instrumentationContextData(), _slicer(NULL), _taskReductions(),
                                 _notifyCopy( NULL ), _notifyThread( NULL ), _remoteAddr( NULL ), _callback(0), _arguments(0),
                                 _submittedWDs( NULL ), _reachedTaskwait( false ), _schedPredecessorLocs(),
                                 _mcontrol( this, numCopies )
                                 {
                                     _devices = new DeviceData*[1];
                                     _devices[0] = device;
                                    _flags.is_final = 0;
                                    _flags.is_submitted = false;
                                    _flags.is_recoverable = false;
                                    _flags.is_invalid = false;
                                    if ( copies != NULL ) {
                                       for ( unsigned int i = 0; i < numCopies; i += 1 ) {
                                          copies[i].setHostBaseAddress( 0 );
                                          copies[i].setRemoteHost( false );
                                       }
                                    }
                                    for (unsigned int __i=0; __i<8;__i+=1) {
                                       _schedValues[__i]=-1;
                                    }
                                 }

inline WorkDescriptor::WorkDescriptor ( const WorkDescriptor &wd, DeviceData **devs, CopyData * copies, void *data, const char *description )
                               : _id( sys.getWorkDescriptorId() ), _hostId( 0 ), _components( 0 ),
                                 _componentsSyncCond( EqualConditionChecker<int>(&_components.override(), 0 ) ), _parent(NULL), _forcedParent(wd._forcedParent),
                                 _data_size( wd._data_size ), _data_align( wd._data_align ), _data ( data ), _totalSize(0),
                                 _wdData ( NULL ), _scheduleData( NULL ),
                                 _flags(), _tiedTo ( wd._tiedTo ), _tiedToLocation( wd._tiedToLocation ),
                                 _state ( INIT ), _syncCond( NULL ), _myQueue ( NULL ), _depth ( wd._depth ),
                                 _numDevices ( wd._numDevices ), _devices ( devs ), _activeDeviceIdx( wd._numDevices == 1 ? 0 : wd._numDevices ),
#ifdef GPU_DEV
                                 _cudaStreamIdx( wd._cudaStreamIdx ),
#endif
                                 _numCopies( wd._numCopies ), _copies( wd._numCopies == 0 ? NULL : copies ), _paramsSize( wd._paramsSize ),
                                 _versionGroupId( wd._versionGroupId ), _executionTime( wd._executionTime ),
                                 _estimatedExecTime( wd._estimatedExecTime ), _doSubmit(NULL), _doWait(),
                                 _depsDomain( sys.getDependenciesManager()->createDependenciesDomain() ),
                                 _translateArgs( wd._translateArgs ),
                                 _priority( wd._priority ), _commutativeOwnerMap(NULL), _commutativeOwners(NULL),
                                 _copiesNotInChunk( wd._copiesNotInChunk), _description(description), _instrumentationContextData(), _slicer(wd._slicer), _taskReductions(),
                                 _notifyCopy( NULL ), _notifyThread( NULL ), _remoteAddr( NULL ), _callback(0), _arguments(0),
                                 _submittedWDs( NULL ), _reachedTaskwait( false ), _schedPredecessorLocs(),
                                 _mcontrol( this, wd._numCopies )
                                 {
                                    if ( wd._parent != NULL ) wd._parent->addWork(*this);
                                    _flags.is_final = wd._flags.is_final;
                                    _flags.is_ready = false;
                                    _flags.to_tie = wd._flags.to_tie;
                                    _flags.is_submitted = false;
                                    _flags.is_implicit = wd._flags.is_implicit;
                                    _flags.is_recoverable = wd._flags.is_recoverable;
                                    _flags.is_invalid = false;
                                    _flags.is_runtime_task = wd._flags.is_runtime_task;

                                    _mcontrol.preInit();
                                    for (unsigned int __i=0; __i<8;__i+=1) {
                                       _schedValues[__i]=-1;
                                    }
                                 }

inline WorkDescriptor::~WorkDescriptor()
{
    void *chunkLower = ( void * ) this;
    void *chunkUpper = ( void * ) ( (char *) this + _totalSize );

    for ( unsigned char i = 0; i < _numDevices; i++ ) delete _devices[i];

    //! Delete device vector
    if ( ( (void*)_devices < chunkLower) || ( (void *) _devices > chunkUpper ) ) {
       delete[] _devices;
    }

    //! Delete Dependence Domain
    delete _depsDomain;

    //! Delete internal data (if any)
    union { char* p; intptr_t i; } u = { (char*)_wdData };
    bool internalDataOwned = (u.i & 1);
    // Clear the own status if set
    u.i &= ~(intptr_t)1;

    if (internalDataOwned
            && (( (void*)u.p < chunkLower) || ( (void *) u.p > chunkUpper ) ))
       delete[] u.p;

    if (_copiesNotInChunk)
        delete[] _copies;
}

/* DeviceData inlined functions */
inline DeviceData::work_fct DeviceData::getWorkFct() const { return _work; }
inline const Device * DeviceData::getDevice () const { return _architecture; }
inline bool DeviceData::isCompatible ( const Device &arch ) { return _architecture == &arch; }

/* WorkDescriptor inlined functions */
inline bool WorkDescriptor::started ( void ) const { return (( _state != INIT ) && (_state != START)); }
inline bool WorkDescriptor::initialized ( void ) const { return ( _state != INIT ) ; }

inline size_t WorkDescriptor::getDataSize () const { return _data_size; }

inline size_t WorkDescriptor::getDataAlignment () const { return _data_align; }

inline void WorkDescriptor::setTotalSize ( size_t size ) { _totalSize = size; }

inline WorkDescriptor * WorkDescriptor::getParent() const { return _parent!=NULL?_parent:_forcedParent ; }
inline void WorkDescriptor::forceParent ( WorkDescriptor * p ) { _forcedParent = p; }

inline WDPool * WorkDescriptor::getMyQueue() { return _myQueue; }
inline void WorkDescriptor::setMyQueue ( WDPool * myQ ) { _myQueue = myQ; }

inline bool WorkDescriptor::isEnqueued() { return ( _myQueue != NULL ); }

inline WorkDescriptor & WorkDescriptor::tied () { _flags.to_tie = true; return *this; }

inline WorkDescriptor & WorkDescriptor::tieTo ( BaseThread &thread )
{
   _tiedTo = &thread;
   _flags.to_tie = false;
   return *this;
}

inline WorkDescriptor & WorkDescriptor::tieToLocation ( memory_space_id_t loc ) { _tiedToLocation = loc; _flags.to_tie=false; return *this; }

inline bool WorkDescriptor::isTied() const { return _tiedTo != NULL; }

inline bool WorkDescriptor::isTiedLocation() const { return _tiedToLocation != ( (memory_space_id_t) -1); }

inline BaseThread* WorkDescriptor::isTiedTo() const { return _tiedTo; }

inline memory_space_id_t WorkDescriptor::isTiedToLocation() const { return _tiedToLocation; }

inline bool WorkDescriptor::shouldBeTied() const { return _flags.to_tie; }

inline void WorkDescriptor::untie() { _tiedTo = NULL; _flags.to_tie = false; }

inline void WorkDescriptor::untieLocation() { _tiedToLocation = ( (memory_space_id_t) -1 ); _flags.to_tie = false; }

inline void WorkDescriptor::setData ( void *wdata ) { _data = wdata; }

inline void * WorkDescriptor::getData () const { return _data; }

inline bool WorkDescriptor::isReady () const { return _flags.is_ready; }

inline void WorkDescriptor::setBlocked () {
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t Keys  = ID->getEventKey("wd-blocked"); )
   NANOS_INSTRUMENT ( if ( _flags.is_ready ) { )
   NANOS_INSTRUMENT ( nanos_event_value_t Values = (nanos_event_value_t) this; )
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &Keys, &Values); )
   NANOS_INSTRUMENT ( } )
   _flags.is_ready = false;
}

inline void WorkDescriptor::setReady ()
{
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t Keys  = ID->getEventKey("wd-ready"); )
   NANOS_INSTRUMENT ( if ( !_flags.is_ready ) { )
   NANOS_INSTRUMENT ( nanos_event_value_t Values = (nanos_event_value_t) this; )
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &Keys, &Values); )
   NANOS_INSTRUMENT ( } )

   _flags.is_ready = true;
}

inline bool WorkDescriptor::isFinal () const { return _flags.is_final; }

inline void WorkDescriptor::setFinal ( bool value ) { _flags.is_final = value; }

inline GenericSyncCond * WorkDescriptor::getSyncCond() { return _syncCond; }

inline void WorkDescriptor::setSyncCond( GenericSyncCond * syncCond ) { _syncCond = syncCond; }

inline void WorkDescriptor::setDepth ( int l ) { _depth = l; }

inline unsigned WorkDescriptor::getDepth() const { return _depth; }

inline DeviceData & WorkDescriptor::getActiveDevice () const { return *_devices[_activeDeviceIdx]; }

inline bool WorkDescriptor::hasActiveDevice() const { return _activeDeviceIdx != _numDevices; }

inline void WorkDescriptor::setActiveDeviceIdx( unsigned char idx ) { _activeDeviceIdx = idx; }

inline unsigned char WorkDescriptor::getActiveDeviceIdx() const { return _activeDeviceIdx; }

#ifdef GPU_DEV
inline void WorkDescriptor::setCudaStreamIdx( int idx ) { _cudaStreamIdx = idx; }

inline int WorkDescriptor::getCudaStreamIdx() const { return _cudaStreamIdx; }
#endif

inline void WorkDescriptor::setInternalData ( void *data, bool ownedByWD ) {
    union { void* p; intptr_t i; } u = { data };
    // Set the own status
    u.i |= int( ownedByWD );

    _wdData = u.p;
}

inline void * WorkDescriptor::getInternalData () const {
    union { void* p; intptr_t i; } u = { _wdData };

    // Clear the own status if set
    u.i &= ~(intptr_t)1;

    return u.p;
}

inline void WorkDescriptor::setSchedulerData ( ScheduleWDData * data, bool ownedByWD ) {
    fatal_cond( _scheduleData != NULL, "Trying to change the scheduler data of a WD that already has one" );

    union { ScheduleWDData * p; intptr_t i; } u = { data };
    // Set the own status
    u.i |= int( ownedByWD );

    _scheduleData = u.p;
}

inline ScheduleWDData * WorkDescriptor::getSchedulerData () const {
    union {ScheduleWDData* p; intptr_t i; } u = { _scheduleData };

    // Clear the own status if set
    u.i &= ~(intptr_t)1;

    return u.p;
}

inline void WorkDescriptor::setTranslateArgs( nanos_translate_args_t translateArgs ) { _translateArgs = translateArgs; }

inline nanos_translate_args_t WorkDescriptor::getTranslateArgs() const { return _translateArgs; }

//inline nanos_translate_args_t WorkDescriptor::getTranslateArgs() { return _translateArgs; }
inline int WorkDescriptor::getNUMANode() const
{
   return _numaNode;
}

inline void WorkDescriptor::setNUMANode( int node )
{
   _numaNode = node;
}

inline unsigned int WorkDescriptor::getNumDevices ( void ) const { return _numDevices; }

inline DeviceData ** WorkDescriptor::getDevices ( void ) const { return _devices; }

inline void WorkDescriptor::clear () { /*_parent = NULL;*/ }

inline size_t WorkDescriptor::getNumCopies() const { return _numCopies; }

inline CopyData * WorkDescriptor::getCopies() const { return _copies; }

inline size_t WorkDescriptor::getParamsSize() const { return _paramsSize; }

inline unsigned long WorkDescriptor::getVersionGroupId( void ) { return _versionGroupId; }

inline void WorkDescriptor::setVersionGroupId( unsigned long id ) { _versionGroupId = id; }

inline double WorkDescriptor::getExecutionTime() const { return _executionTime; }

inline double WorkDescriptor::getEstimatedExecutionTime() const { return _estimatedExecTime; }

inline void WorkDescriptor::setEstimatedExecutionTime( double time ) { _estimatedExecTime = time; }

inline DOSubmit * WorkDescriptor::getDOSubmit() { return _doSubmit; }

inline int WorkDescriptor::getNumDepsPredecessors() { return ( _doSubmit == NULL ? 0 : _doSubmit->numPredecessors() ); }

inline bool WorkDescriptor::hasDepsPredecessors() { return ( _doSubmit == NULL ? false : ( _doSubmit->numPredecessors() != 0 ) ); }

inline void WorkDescriptor::submitWithDependencies( WorkDescriptor &wd, size_t numDeps, DataAccess* deps )
{
   wd._doSubmit = NEW DOSubmit();
   wd._doSubmit->setWD(&wd);

   // Defining call back (cb)
   SchedulePolicySuccessorFunctor cb( *sys.getDefaultSchedulePolicy() );

   initCommutativeAccesses( wd, numDeps, deps );

   _depsDomain->submitDependableObject( *(wd._doSubmit), numDeps, deps, &cb );
   if ( sys._preSchedule ) {
      sys._slots[wd._doSubmit->getNum()].insert(&wd);
   }

}

inline void WorkDescriptor::waitOn( size_t numDeps, DataAccess* deps )
{
   _doWait->setWD(this);
   _depsDomain->submitDependableObject( *_doWait, numDeps, deps );
   _mcontrol.synchronize( numDeps, deps );
}

class DOIsSchedulable : public DependableObjectPredicate
{
   BaseThread &    _thread;

   public:
      DOIsSchedulable(BaseThread &thread) : DependableObjectPredicate(),_thread(thread) { }
      ~DOIsSchedulable() {}

      bool operator() ( DependableObject &obj )
      {
         WD *wd = (WD *)obj.getRelatedObject();
         // FIXME: The started condition here ensures that doWait objects are not released as
         // they do not work properly if there is no dependenceSatisfied called before
         return (wd != NULL) && Scheduler::checkBasicConstraints(*wd,_thread) && !wd->started() ;
      }
};

inline WorkDescriptor * WorkDescriptor::getImmediateSuccessor ( BaseThread &thread )
{
   if ( _doSubmit == NULL || !sys.isImmediateSuccessorEnabled() ) return NULL;
   else {
      DOIsSchedulable predicate( thread );
      DependableObject * found = _doSubmit->releaseImmediateSuccessor( predicate, thread.keepWDDeps() );
      if ( found ) {
         WD *successor = found->getWD();
         //successor->predecessorFinished( this );
         successor->_mcontrol.preInit();
         return successor;
      } else {
         return NULL;
      }
   }
}

inline void WorkDescriptor::workFinished(WorkDescriptor &wd)
{
   if ( wd._doSubmit != NULL ){
      wd._doSubmit->finished();
      delete wd._doSubmit;
      wd._doSubmit = NULL;
   }
}

inline void WorkDescriptor::releaseInputDependencies()
{
   if ( _doSubmit != NULL ){
      _doSubmit->releaseReadDependencies();
   }
}

inline DependenciesDomain & WorkDescriptor::getDependenciesDomain()
{
   return *_depsDomain;
}


inline InstrumentationContextData * WorkDescriptor::getInstrumentationContextData( void ) { return &_instrumentationContextData; }

inline bool WorkDescriptor::isSubmitted() const { return _flags.is_submitted; }
inline void WorkDescriptor::submitted()  { _flags.is_submitted = true; }

inline bool WorkDescriptor::isConfigured ( void ) const { return _flags.is_configured; }
inline void WorkDescriptor::setConfigured ( bool value ) { _flags.is_configured = value; }

inline void WorkDescriptor::setPriority( WorkDescriptor::PriorityType priority )
{
   _priority = priority;
   if ( _myQueue ) myThread->getTeam()->getSchedulePolicy().reorderWD( myThread, this );
}
inline WorkDescriptor::PriorityType WorkDescriptor::getPriority() const { return _priority; }

inline void WorkDescriptor::releaseCommutativeAccesses()
{
   if ( _commutativeOwners == NULL ) return;
   const size_t n = _commutativeOwners->size();
   for ( size_t i = 0; i < n; i++ )
      *(*_commutativeOwners)[i] = NULL;
}

inline void WorkDescriptor::setImplicit( bool b )
{
   //! Set implicit flag to parameter value
   _flags.is_implicit = b;

   //! Unset parent to free current Work Descriptor from hierarchy
   if ( _parent != NULL ) {
      _parent->exitWork(*this);
      _parent = NULL;
   }
}

inline bool WorkDescriptor::isImplicit( void ) { return _flags.is_implicit; }

inline void WorkDescriptor::setRuntimeTask( bool b )
{
  _flags.is_runtime_task = b;
}

inline bool WorkDescriptor::isRuntimeTask( void ) const { return _flags.is_runtime_task; }

inline const char * WorkDescriptor::getDescription ( void ) const  { return _description; }

inline void WorkDescriptor::addWork ( WorkDescriptor &work )
{
   _components++;
   work.addToGroup( *this );
}

inline void WorkDescriptor::addToGroup ( WorkDescriptor &parent )
{
   if ( _parent == NULL ) _parent = &parent;
   else fatal("WorkDescriptor: Trying to add a second parent");
}

inline Slicer * WorkDescriptor::getSlicer ( void ) const
{
   return _slicer;
}

inline void WorkDescriptor::setSlicer ( Slicer *slicer )
{
    _slicer = slicer;
}

inline bool WorkDescriptor::dequeue ( WorkDescriptor **slice )
{
   if ( _slicer ) return _slicer->dequeue( this, slice );
   else {
      *slice = this;
      return true;
   }
}

inline void WorkDescriptor::convertToRegularWD()
{
   _slicer = NULL;
}

inline void WorkDescriptor::copyReductions(WorkDescriptor *parent)
{
	_taskReductions = parent->_taskReductions;
}

inline void WorkDescriptor::setId( unsigned int id ) {
   _id = id;
}

inline void WorkDescriptor::setRemoteAddr( void const *addr ) {
   _remoteAddr = addr;
}

inline void const *WorkDescriptor::getRemoteAddr() const {
   return _remoteAddr;
}

inline bool WorkDescriptor::setInvalid ( bool flag )
{
   if (_flags.is_invalid != flag) {
      _flags.is_invalid = flag;

      /*
       * Note: At this time, do not take any action if the task is invalid and it has
       * no parent. There could be some special cases where it does not imply a fatal
       * error.
       */
      if (_flags.is_invalid && !_flags.is_recoverable) {
         if (_parent == NULL)
            /*
             *  If no invalidity propagation is possible (this task is the root in some way)
             *  return that no recoverable task was found at this point, so any action can be taken
             *  accordingly.
             */
            return false;
         else if (!_parent->_flags.is_invalid) {
            // If this task is not recoverable, propagate invalidation to its parent.
            return _parent->setInvalid(true);
            return true;
         }
      }
   }
   return true;
}

inline bool WorkDescriptor::isInvalid() const { return _flags.is_invalid; }

inline void WorkDescriptor::setRecoverable( bool flag ) { _flags.is_recoverable = flag; }

inline bool WorkDescriptor::isRecoverable() const { return _flags.is_recoverable; }

inline void WorkDescriptor::setCriticality ( int cr ) { _criticality = cr; }

inline int  WorkDescriptor::getCriticality () const { return _criticality; }

inline void WorkDescriptor::setCallback ( void *cb ) { _callback = cb; }

inline void WorkDescriptor::setArguments ( void *a ) { _arguments = a; }

} // namespace nanos

#endif
