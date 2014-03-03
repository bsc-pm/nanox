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

#include "workdescriptor.hpp"
#include "schedule.hpp"
#include "processingelement.hpp"
#include "basethread.hpp"
#include "debug.hpp"
#include "schedule.hpp"
#include "system.hpp"
#include "os.hpp"
#include "synchronizedcondition.hpp"

using namespace nanos;


/*inline*/ WorkDescriptor::WorkDescriptor ( int ndevices, DeviceData **devs, size_t data_size, size_t data_align, void *wdata,
                                 size_t numCopies, CopyData *copies, nanos_translate_args_t translate_args, char *description )
                               : _id( sys.getWorkDescriptorId() ), _components( 0 ), 
                                 _componentsSyncCond( EqualConditionChecker<int>( &_components.override(), 0 ) ), _parent(NULL), _forcedParent(NULL),
                                 _data_size ( data_size ), _data_align( data_align ),  _data ( wdata ), _totalSize(0),
                                 _wdData ( NULL ), _flags(), _tiedTo ( NULL ),
                                 _state( INIT ), _syncCond( NULL ),  _myQueue ( NULL ), _depth ( 0 ),
                                 _numDevices ( ndevices ), _devices ( devs ), _activeDeviceIdx( ndevices == 1 ? 0 : ndevices ),
                                 _numCopies( numCopies ), _copies( copies ), _paramsSize( 0 ),
                                 _versionGroupId( 0 ), _executionTime( 0.0 ), _estimatedExecTime( 0.0 ),
                                 _doSubmit(NULL), _doWait(), _depsDomain( sys.getDependenciesManager()->createDependenciesDomain() ), 
                                 _directory(NULL), _implicit(false), _translateArgs( translate_args ),
                                 _priority( 0 ), _commutativeOwnerMap(NULL), _commutativeOwners(NULL), _wakeUpQueue( UINT_MAX ),
                                 _copiesNotInChunk(false), _description(description), _instrumentationContextData()
                                 {
                                    _flags.is_final = 0;
                                    _flags.is_submitted = false;
                                 }

/*inline*/ WorkDescriptor::WorkDescriptor ( DeviceData *device, size_t data_size, size_t data_align, void *wdata,
                                 size_t numCopies, CopyData *copies, nanos_translate_args_t translate_args, char *description )
                               : _id( sys.getWorkDescriptorId() ), _components( 0 ), 
                                 _componentsSyncCond( EqualConditionChecker<int>( &_components.override(), 0 ) ), _parent(NULL), _forcedParent(NULL),
                                 _data_size ( data_size ), _data_align ( data_align ), _data ( wdata ), _totalSize(0),
                                 _wdData ( NULL ), _flags(), _tiedTo ( NULL ),
                                 _state( INIT ), _syncCond( NULL ), _myQueue ( NULL ), _depth ( 0 ),
                                 _numDevices ( 1 ), _devices ( NULL ), _activeDeviceIdx( 0 ),
                                 _numCopies( numCopies ), _copies( copies ), _paramsSize( 0 ),
                                 _versionGroupId( 0 ), _executionTime( 0.0 ), _estimatedExecTime( 0.0 ), 
                                 _doSubmit(NULL), _doWait(), _depsDomain( sys.getDependenciesManager()->createDependenciesDomain() ),
                                 _directory(NULL), _implicit(false), _translateArgs( translate_args ),
                                 _priority( 0 ),  _commutativeOwnerMap(NULL), _commutativeOwners(NULL),
                                 _wakeUpQueue( UINT_MAX ), _copiesNotInChunk(false), _description(description), _instrumentationContextData()
                                 {
                                    _devices = new DeviceData*[1];
                                    _devices[0] = device;
                                    _flags.is_final = 0;
                                    _flags.is_submitted = false;
                                 }

/*inline*/ WorkDescriptor::WorkDescriptor ( const WorkDescriptor &wd, DeviceData **devs, CopyData * copies, void *data, char *description )
                               : _id( sys.getWorkDescriptorId() ), _components( 0 ), 
                                 _componentsSyncCond( EqualConditionChecker<int>(&_components.override(), 0 ) ), _parent(NULL), _forcedParent(wd._forcedParent),
                                 _data_size( wd._data_size ), _data_align( wd._data_align ), _data ( data ), _totalSize(0),
                                 _wdData ( NULL ), _flags(), _tiedTo ( wd._tiedTo ),
                                 _state ( INIT ), _syncCond( NULL ), _myQueue ( NULL ), _depth ( wd._depth ),
                                 _numDevices ( wd._numDevices ), _devices ( devs ), _activeDeviceIdx( wd._numDevices == 1 ? 0 : wd._numDevices ),
                                 _numCopies( wd._numCopies ), _copies( wd._numCopies == 0 ? NULL : copies ), _paramsSize( wd._paramsSize ),
                                 _versionGroupId( wd._versionGroupId ), _executionTime( wd._executionTime ),
                                 _estimatedExecTime( wd._estimatedExecTime ), _doSubmit(NULL), _doWait(),
                                 _depsDomain( sys.getDependenciesManager()->createDependenciesDomain() ),
                                 _directory(NULL), _implicit( wd._implicit ),_translateArgs( wd._translateArgs ),
                                 _priority( wd._priority ), _commutativeOwnerMap(NULL), _commutativeOwners(NULL),
                                 _wakeUpQueue( wd._wakeUpQueue ),
                                 _copiesNotInChunk( wd._copiesNotInChunk), _description(description), _instrumentationContextData()
                                 {
                                    if ( wd._parent != NULL ) wd._parent->addWork(*this);
                                    _flags.is_final = false;
                                    _flags.is_ready = false;
                                    _flags.to_tie = wd._flags.to_tie;
                                    _flags.is_submitted = false;
                                 }


void WorkDescriptor::init ()
{
   if ( _state != INIT ) return;

   ProcessingElement *pe = myThread->runningOn();

   /* Initializing instrumentation context */
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdCreate( this ) );

   _executionTime = ( _numDevices == 1 ? 0.0 : OS::getMonotonicTimeUs() );

   if ( getNumCopies() > 0 ) {
      pe->copyDataIn( *this );
      if ( _translateArgs != NULL ) {
         _translateArgs( _data, this );
      }
   }
   setStart();
}

// That function must be called from the thread it will execute it. This is important
// from the point of view of tiedness and the device activation. Both operation will
// involves current thread / pe
void WorkDescriptor::start (ULTFlag isUserLevelThread, WorkDescriptor *previous)
{
   ensure ( _state == START , "Trying to start a wd twice or trying to start an uninitialized wd");

   ProcessingElement *pe = myThread->runningOn();


   // If there are no active device, choose a compatible one
   if ( _activeDeviceIdx == _numDevices ) activateDevice ( pe->getDeviceType() );

   // Initializing devices
   _devices[_activeDeviceIdx]->lazyInit( *this, isUserLevelThread, previous );

   // Waiting for copies
   if ( getNumCopies() > 0 ) pe->waitInputs( *this );

   // Tie WD to current thread
   if ( _flags.to_tie ) tieTo( *myThread );

   // Call Programming Model interface .started() method.
   sys.getPMInterface().wdStarted( *this );

   // Setting state to ready
   _state = READY; //! \bug This should disapear when handling properly states as flags (#904)
}

void WorkDescriptor::prepareDevice ()
{
   // Do nothing if there is already an active device
   if ( _activeDeviceIdx != _numDevices ) return;

   if ( _numDevices == 1 ) {
      _activeDeviceIdx = 0;
      return;
   }

   // Choose between the supported devices
   message("No active device --> selecting one");
   _activeDeviceIdx = _numDevices - 1;
}

DeviceData & WorkDescriptor::activateDevice ( const Device &device )
{
   if ( _activeDeviceIdx != _numDevices ) {
      ensure( _devices[_activeDeviceIdx]->isCompatible( device ),"Bogus double device activation" );
      return *_devices[_activeDeviceIdx];
   }
   unsigned i = _numDevices;
   for ( i = 0; i < _numDevices; i++ ) {
      if ( _devices[i]->isCompatible( device ) ) {
         _activeDeviceIdx = i;
         break;
      }
   }

   ensure( i < _numDevices, "Did not find requested device in activation" );

   return *_devices[_activeDeviceIdx];
}

DeviceData & WorkDescriptor::activateDevice ( unsigned int deviceIdx )
{
   ensure( _numDevices > deviceIdx, "The requested device number does not exist" );

   _activeDeviceIdx = deviceIdx;

   return *_devices[_activeDeviceIdx];
}

bool WorkDescriptor::canRunIn( const Device &device ) const
{
   if ( _activeDeviceIdx != _numDevices ) return _devices[_activeDeviceIdx]->isCompatible( device );

   unsigned int i;
   for ( i = 0; i < _numDevices; i++ ) {
      if ( _devices[i]->isCompatible( device ) ) {
         return true;
      }
   }

   return false;
}

bool WorkDescriptor::canRunIn ( const ProcessingElement &pe ) const
{
   if ( started() && !pe.supportsUserLevelThreads() ) return false;
   return canRunIn( pe.getDeviceType() );
}

void WorkDescriptor::submit( bool force_queue )
{
   Scheduler::submit(*this, force_queue );
} 

void WorkDescriptor::finish ()
{
   // At that point we are ready to copy data out
   if ( getNumCopies() > 0 ) {
      ProcessingElement *pe = myThread->runningOn();
      pe->copyDataOut( *this );
   }

   // Getting execution time
   _executionTime = ( _numDevices == 1 ? 0.0 : OS::getMonotonicTimeUs() - _executionTime );
}

void WorkDescriptor::done ()
{
   // Releasing commutative accesses
   releaseCommutativeAccesses(); 

   // Executing programming model specific finalization
   sys.getPMInterface().wdFinished( *this );

   // Notifying parent we have finished ( dependence's relationships )
   this->getParent()->workFinished( *this );

//! \bug FIXME: This instrumentation phase has been commented due may cause raises when creating the events
#if 0
   NANOS_INSTRUMENT ( static Instrumentation *instr = sys.getInstrumentation(); )
#endif

   // Waiting for children (just to keep structures)
   if ( _components != 0 ) waitCompletion();

   // Notifying parent about current WD finalization
   if ( _parent != NULL ) {
      _parent->exitWork(*this);
#if 0
      NANOS_INSTRUMENT ( if ( !_parent->isReady()) { )
      NANOS_INSTRUMENT ( nanos_event_id_t id = ( ((nanos_event_id_t) getId()) << 32 ) + _parent->getId(); )
      NANOS_INSTRUMENT ( instr->raiseOpenPtPEvent ( NANOS_WAIT, id, 0, 0 );)
      NANOS_INSTRUMENT ( instr->createDeferredPtPEnd ( *_parent, NANOS_WAIT, id, 0, 0 ); )
      NANOS_INSTRUMENT ( } )
#endif
      _parent = NULL;
   }
}

void WorkDescriptor::prepareCopies()
{
   for (unsigned int i = 0; i < _numCopies; i++ ) {
      _paramsSize += _copies[i].getSize();

      if ( _copies[i].isPrivate() )
         //jbueno new API _copies[i].setAddress( ( (uint64_t)_copies[i].getAddress() - (unsigned long)_data ) );
         _copies[i].setBaseAddress( (void *) ( (uint64_t )_copies[i].getBaseAddress() - (unsigned long)_data ) );
   }
}

void WorkDescriptor::initCommutativeAccesses( WorkDescriptor &wd, size_t numDeps, DataAccess* deps )
{
   size_t numCommutative = 0;

   for ( size_t i = 0; i < numDeps; i++ )
      if ( deps[i].isCommutative() )
         ++numCommutative;

   if ( numCommutative == 0 )
      return;
   if (wd._commutativeOwners == NULL) wd._commutativeOwners = NEW WorkDescriptorPtrList();
   wd._commutativeOwners->reserve(numCommutative);

   for ( size_t i = 0; i < numDeps; i++ ) {
      if ( !deps[i].isCommutative() )
         continue;

      if ( _commutativeOwnerMap == NULL ) _commutativeOwnerMap = NEW CommutativeOwnerMap();

      // Lookup owner in map in parent WD
      CommutativeOwnerMap::iterator iter = _commutativeOwnerMap->find( deps[i].getDepAddress() );

      if ( iter != _commutativeOwnerMap->end() ) {
         // Already in map => insert into owner list in child WD
         wd._commutativeOwners->push_back( iter->second.get() );
      }
      else {
         // Not in map => allocate new owner pointer container and insert
         std::pair<CommutativeOwnerMap::iterator, bool> ret =
               _commutativeOwnerMap->insert( std::make_pair( deps[i].getDepAddress(),
                                                            TR1::shared_ptr<WorkDescriptor *>( NEW WorkDescriptor *(NULL) ) ) );

         // Insert into owner list in child WD
         wd._commutativeOwners->push_back( ret.first->second.get() );
      }
   }
}

bool WorkDescriptor::tryAcquireCommutativeAccesses()
{
   if ( _commutativeOwners == NULL ) return true;

   const size_t n = _commutativeOwners->size();
   for ( size_t i = 0; i < n; i++ ) {

      WorkDescriptor *owner = *(*_commutativeOwners)[i];

      if ( owner == this )
         continue;

      if ( owner == NULL &&
           nanos::compareAndSwap( (void **) (*_commutativeOwners)[i], (void *) NULL, (void *) this ) )
         continue;

      // Failed to obtain exclusive access to all accesses, release the obtained ones

      for ( ; i > 0; i-- )
         *(*_commutativeOwners)[i-1] = NULL;

      return false;
   }
   return true;
} 

void WorkDescriptor::setCopies(size_t numCopies, CopyData * copies)
{
    ensure(_numCopies == 0, "This WD already had copies. Overriding them is not possible");
    ensure((numCopies == 0) == (copies == NULL), "Inconsistency between copies and number of copies");

    _numCopies = numCopies;

    _copies = NEW CopyData[numCopies];
    _copiesNotInChunk = true;

    // Keep a copy of the copy descriptors
    std::copy(copies, copies + numCopies, _copies);

    for (unsigned int i = 0; i < numCopies; ++i)
    {
        int num_dimensions = copies[i].dimension_count;
        if ( num_dimensions > 0 ) {
            nanos_region_dimension_internal_t* copy_dims = NEW nanos_region_dimension_internal_t[num_dimensions];
            std::copy(copies[i].dimensions, copies[i].dimensions + num_dimensions, copy_dims);
            _copies[i].dimensions = copy_dims;
        } else {
            _copies[i].dimensions = NULL;
        }
    }
}

void WorkDescriptor::waitCompletion( bool avoidFlush )
{
   _componentsSyncCond.waitConditionAndSignalers();
   if ( _directory != NULL && !avoidFlush ) _directory->synchronizeHost();
}

void WorkDescriptor::exitWork ( WorkDescriptor &work )
{
   _componentsSyncCond.reference();
   int componentsLeft = --_components;
   //! \note It seems that _syncCond.check() generates a race condition here?
   if (componentsLeft == 0) _componentsSyncCond.signal();
   _componentsSyncCond.unreference();
}


