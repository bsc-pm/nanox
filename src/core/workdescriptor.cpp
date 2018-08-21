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

#include "workdescriptor.hpp"
#include "schedule.hpp"
#include "processingelement.hpp"
#include "basethread.hpp"
#include "debug.hpp"
#include "schedule.hpp"
#include "system.hpp"
#include "os.hpp"
#include "synchronizedcondition.hpp"
#include "basethread.hpp"

using namespace nanos;

void WorkDescriptor::init ()
{
   if ( _state != INIT ) return;

   ProcessingElement *pe = myThread->runningOn();

   /* Initializing instrumentation context */
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdCreate( this ) );

   _executionTime = ( sys.getDefaultSchedulePolicy()->isCheckingWDExecTime() ? OS::getMonotonicTimeUs() : 0.0 );

   if ( getNumCopies() > 0 ) {
      pe->copyDataIn( *this );
      //this->notifyCopy();

      if ( _translateArgs != NULL ) {
         _translateArgs( _data, this );
      }
   }
   _state = WorkDescriptor::START;
}

void WorkDescriptor::initWithPE ( ProcessingElement &pe )
{
   if ( _state != INIT ) return;

   /* Initializing instrumentation context */
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdCreate( this ) );

   //message("init wd " << getId() );
   //if ( getNewDirectory() == NULL )
   //   initNewDirectory();
   //getNewDirectory()->setParent( ( getParent() != NULL ) ? getParent()->getNewDirectory() : NULL );

   if ( getNumCopies() > 0 ) {

      //CopyData *copies = getCopies();
      //for ( unsigned int i = 0; i < getNumCopies(); i++ ) {
      //   CopyData & cd = copies[i];
      //   if ( !cd.isPrivate() ) {
      //      //message("[n:" << sys.getNetwork()->getNodeNum() << "] WD "<< getId() << " init DA["<< i << "]: addr is " << (void *) cd.getDataAccess()->address );
      //      //DataAccess d( cd.getDataAccess()->address, cd.getDataAccess()->flags.input ,cd.getDataAccess()->flags.output, cd.getDataAccess()->flags.can_rename,
      //      //   cd.getDataAccess()->flags.commutative, cd.getDataAccess()->dimension_count, cd.getDataAccess()->dimensions);
      //      //  Region reg = NewRegionDirectory::build_region( d );
      //      //  message("region is " << reg);
      //      //  getNewDirectory()->registerAccess( reg, cd.isInput(), cd.isOutput(), pe->getMemorySpaceId() );
      //   }
      //}

      _notifyThread = pe.getFirstThread();
      pe.copyDataIn( *this );
      //this->notifyCopy();

      if ( _translateArgs != NULL ) {
         _translateArgs( _data, this );
      }
   }
   _state = WorkDescriptor::START;
}


void WorkDescriptor::notifyCopy()
{
   if ( _notifyCopy != NULL ) {
      //std::cerr << " WD " << getId() << " GONNA CALL THIS SHIT IF POSSIBLE: " << (void *)_notifyCopy << " ARG IS THD "<< _notifyThread->getId() << std::endl;
      _notifyCopy( *this, *_notifyThread );
   }
}

// That function must be called from the thread it will execute it. This is important
// from the point of view of tiedness and the device activation. Both operations will
// involve current thread / pe
void WorkDescriptor::start (ULTFlag isUserLevelThread, WorkDescriptor *previous)
{
   ensure ( _state == START , "Trying to start a wd twice or trying to start an uninitialized wd");

   // If there is no active device, choose a compatible one
   ProcessingElement *pe = myThread->runningOn();
   if ( _activeDeviceIdx == _numDevices ) activateDevice ( *pe->getActiveDevice() );

   // Initializing devices
   _devices[_activeDeviceIdx]->lazyInit( *this, isUserLevelThread, previous );

   ensure ( _activeDeviceIdx != _numDevices, "This WD has no active device. If you are using 'implements' feature, please use versioning scheduler." );

   // Waiting for copies
   if ( getNumCopies() > 0 ) {
      pe->waitInputs( *this );
   }

   // Tie WD to current thread
   if ( _flags.to_tie ) tieTo( *myThread );

   // Call Programming Model interface .started() method.
   sys.getPMInterface().wdStarted( *this );

   // Setting state to ready
   _state = READY; //! \bug This should disapear when handling properly states as flags (#904)
   _mcontrol.setCacheMetaData();
}


void WorkDescriptor::preStart (ULTFlag isUserLevelThread, WorkDescriptor *previous)
{
   ensure ( _state == START , "Trying to start a wd twice or trying to start an uninitialized wd");

   ProcessingElement *pe = myThread->runningOn();

   // If there is no active device, choose a compatible one
   if ( _activeDeviceIdx == _numDevices ) activateDevice ( *pe->getActiveDevice() );

   // Initializing devices
   _devices[_activeDeviceIdx]->lazyInit( *this, isUserLevelThread, previous );

   _mcontrol.setCacheMetaData();

}

bool WorkDescriptor::isInputDataReady() {
   ProcessingElement *pe = myThread->runningOn();
   bool result = false;

   // Test if copies have completed
   if ( getNumCopies() > 0 ) {
      result = pe->testInputs( *this );
   } else {
      result = true;
   }

   if ( result ) {
      // Tie WD to current thread
      if ( _flags.to_tie ) tieTo( *myThread );

      // Call Programming Model interface .started() method.
      sys.getPMInterface().wdStarted( *this );

      // Setting state to ready
      setReady();
      //_mcontrol.setCacheMetaData();
   }
   return result;
}


void WorkDescriptor::prepareDevice ()
{
   // TODO: This function is never called, so we should remove it
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
       if (_devices[i]->isCompatible( device )){
            return true;
       }
   }
   return false;
}

bool WorkDescriptor::canRunIn ( const ProcessingElement &pe ) const
{
   warning("WorkDescriptor::canRunIn(ProcessingElement) is deprecated. Use PE::canRun(WD) instead.");
   return pe.canRun( *this );
}

void WorkDescriptor::submit( bool force_queue )
{
   _mcontrol.preInit();

   if ( _slicer ) {
      _slicer->submit(*this);
   } else {
      Scheduler::submit(*this, force_queue );
   }
}

void WorkDescriptor::submitOutputCopies ()
{
   if ( getNumCopies() > 0 ) {
      _mcontrol.copyDataOut( MemController::WRITE_BACK );
   }
}

void WorkDescriptor::waitOutputCopies ()
{
   if ( getNumCopies() > 0 ) {
      while ( !_mcontrol.isOutputDataReady( *this ) ) {
         myThread->processTransfers();
      }
   }
}

void WorkDescriptor::finish ()
{
   // At that point we are ready to copy data out
   if ( getNumCopies() > 0 ) {
      _mcontrol.copyDataOut( MemController::WRITE_BACK );
      while ( !_mcontrol.isOutputDataReady( *this ) ) {
         myThread->processTransfers();
      }
   }

   // Getting execution time
   _executionTime = ( sys.getDefaultSchedulePolicy()->isCheckingWDExecTime() ? OS::getMonotonicTimeUs() - _executionTime : 0.0 );
}

void WorkDescriptor::preFinish ()
{
   // At that point we are ready to copy data out
   if ( getNumCopies() > 0 ) {
      _mcontrol.copyDataOut( MemController::WRITE_BACK );
   }

   // Getting execution time
   _executionTime = ( sys.getDefaultSchedulePolicy()->isCheckingWDExecTime() ? OS::getMonotonicTimeUs() - _executionTime : 0.0 );
}


bool WorkDescriptor::isOutputDataReady()
{
   // Test if copies have completed
   if ( getNumCopies() > 0 ) {
      return _mcontrol.isOutputDataReady( *this );
   }

   return true;
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

   #ifdef NANOX_TASK_CALLBACK
   typedef void (* notify_t) ( void * );
   notify_t notify = (notify_t) _callback;
   if (notify ) notify(_arguments);
   #endif
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

void WorkDescriptor::notifyOutlinedCompletion()
{
   ensure( isTied(), "Outlined WD completed, but it is untied!");
   _tiedTo->notifyOutlinedCompletionDependent( this );
}
void WorkDescriptor::predecessorFinished( WorkDescriptor *predecessorWd )
{
   //if ( predecessorWd != NULL )
   //{
   //   setMyGraphRepList( predecessorWd->getMyGraphRepList() );
   //}
   //
   //if ( _myGraphRepList == NULL ) {
   //   _myGraphRepList = sys.getGraphRepList();
   //   if ( predecessorWd != NULL ) {
   //      _myGraphRepList.value()->push_back( predecessorWd->getGE() );
   //      predecessorWd->listed();
   //   }
   //}
   //_myGraphRepList.value()->push_back( getGE() );
   //if (predecessorWd != NULL) predecessorWd->listed();

   //*(myThread->_file) << "I'm " << getId() << " : " << getDescription() << " my predecessor " << predecessorWd->getId() << " : " << predecessorWd->getDescription() << " has finished." << std::endl;
   _mcontrol.getInfoFromPredecessor( predecessorWd->_mcontrol );
}

void WorkDescriptor::wgdone()
{
   //if (!_listed)
   //{
   //   if ( _myGraphRepList == NULL ) {
   //      _myGraphRepList = sys.getGraphRepList();
   //   }
   //   _myGraphRepList.value()->push_back( this->getParent()->getGENext() );
   //}
}

//void WorkDescriptor::listed()
//{
//   _listed = true;
//}

void WorkDescriptor::printCopies()
{
      CopyData *copies = getCopies();
      std::cerr << "############################################"<< std::endl;
      for ( unsigned int i = 0; i < getNumCopies(); i++ ) {
         CopyData & cd = copies[i];
         std::cerr << "# Copy "<< i << std::endl << cd << std::endl;
         if ( i+1 < getNumCopies() ) std::cerr << " --------------- "<< std::endl;
      }
      std::cerr << "############################################"<< std::endl;

}
void WorkDescriptor::setNotifyCopyFunc( void (*func)(WD &, BaseThread const&) ) {
   _notifyCopy = func;
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
        _copies[i].setHostBaseAddress( 0 );
        _copies[i].setRemoteHost( false );
    }

   new ( &_mcontrol ) MemController( this, numCopies );
}

void WorkDescriptor::waitCompletion( bool avoidFlush )
{
   sys.preSchedule();
   _reachedTaskwait = true;
   if ( _submittedWDs != NULL && _submittedWDs->size() > 0 ) {
      Scheduler::_submit( &(*_submittedWDs)[0], _submittedWDs->size() );
      delete _submittedWDs;
      _submittedWDs = NULL;
   }
   _depsDomain->finalizeAllReductions();
   _componentsSyncCond.waitConditionAndSignalers();
   if ( !avoidFlush ) {
      _mcontrol.synchronize();
   }
   _reachedTaskwait = false;

   removeAllTaskReductions();

   if ( sys.getPMInterface().isOmpSs() ) {
      myThread->getTeam()->computeVectorReductions();
      myThread->getTeam()->cleanUpReductionList();
   }

   _depsDomain->clearDependenciesDomain();
}

void WorkDescriptor::exitWork ( WorkDescriptor &work )
{
   _componentsSyncCond.reference();
   int componentsLeft = --_components;
   //! \note It seems that _syncCond.check() generates a race condition here?
   if (componentsLeft == 0) _componentsSyncCond.signal();
   _componentsSyncCond.unreference();
}

void WorkDescriptor::registerTaskReduction( void *p_orig, size_t p_size, size_t p_el_size,
      void (*p_init)( void *, void * ), void (*p_reducer)( void *, void * ) )
{
   //! Check if we have registered a reduction with this address
   task_reduction_vector_t::reverse_iterator it;
   for ( it = _taskReductions.rbegin(); it != _taskReductions.rend(); it++) {
      if ( (*it)->has( p_orig) )
      {
    	  return;
      }
   }

   if ( it == _taskReductions.rend() ) {
       //! We must register p_orig as a new reduction
       _taskReductions.push_back(
               new TaskReduction(
            		   p_orig,
					   p_init,
					   p_reducer,
					   p_size,
					   p_el_size,
					   sys.getThreadManager()->getMaxThreads(),
					   myThread->getCurrentWD()->getDepth(),
					   sys._lazyPrivatizationEnabled
					   )
       );
   }
}

void WorkDescriptor::registerFortranArrayTaskReduction( void *p_orig, void *p_dep, size_t array_descriptor_size,
      void (*p_init)( void *, void * ), void (*p_reducer)( void *, void * ), void (*p_reducer_orig_var)( void *, void * ) )
{
   //! Check if we have registered a reduction with this address
   task_reduction_vector_t::reverse_iterator it;
   for ( it = _taskReductions.rbegin(); it != _taskReductions.rend(); it++) {
      if ( (*it)->has( p_dep) ) break;
   }

   if ( it == _taskReductions.rend() ) {
      //! We must register p_orig as a new reduction
     _taskReductions.push_back(
            new TaskReduction(
            		p_orig,
					p_dep,
					p_init,
					p_reducer,
					p_reducer_orig_var,
					array_descriptor_size,
					sys.getThreadManager()->getMaxThreads(),
					myThread->getCurrentWD()->getDepth(),
					sys._lazyPrivatizationEnabled
					)
     );
   }
}

void * WorkDescriptor::getTaskReductionThreadStorage( void *p_addr, size_t id )
{
   //! Check if we have registered a reduction with this address
   task_reduction_vector_t::reverse_iterator it;
   for ( it = _taskReductions.rbegin(); it != _taskReductions.rend(); it++) {
      if((*it)->has( p_addr )) break;
   }

   // If 'p_addr' is not registered as a reduction we should return NULL
   void *storage = NULL;

   if ( it != _taskReductions.rend() ) {
      storage = (*it)->get(id);

      if ( storage == NULL )
         storage = (*it)->allocate(id);

      if ( !(*it)->isInitialized(id) )
         (*it)->initialize(id);
   }
   return storage;
}

void WorkDescriptor::removeAllTaskReductions( void )
{
   task_reduction_vector_t::reverse_iterator it;
   for ( it = _taskReductions.rbegin(); it != _taskReductions.rend(); it++) {
      // Am I the owner of this reduction?
      if (_depth == (*it)->getDepth()) {
         delete (*it);
         _taskReductions.erase( --(it.base()) );
      }
   }
}

TaskReduction * WorkDescriptor::getTaskReduction( const void *p_dep )
{
   // Check if we have registered a reduction with this address
   task_reduction_vector_t::reverse_iterator it;
   for ( it = _taskReductions.rbegin(); it != _taskReductions.rend(); it++) {
	   if ( (*it)->has( p_dep ) ) return (*it);
   }
   return NULL;
}

bool WorkDescriptor::resourceCheck( BaseThread const &thd, bool considerInvalidations ) const {
   return _mcontrol.canAllocateMemory( thd.runningOn()->getMemorySpaceId(), considerInvalidations );
}

//void WorkDescriptor::initMyGraphRepListNoPred( ) {
//   _myGraphRepList = sys.getGraphRepList();
//   _myGraphRepList.value()->push_back( this->getParent()->getGE() );
//   _myGraphRepList.value()->push_back( getGE() );
//}

//void WorkDescriptor::setMyGraphRepList( std::list<GraphEntry *> *myList ) {
//   _myGraphRepList = myList;
//}

//std::list<GraphEntry *> *WorkDescriptor::getMyGraphRepList(  )
//{
//   std::list<GraphEntry *> *myList = NULL;
//   do {
//      myList = _myGraphRepList.value();
//   }
//   while ( ! _myGraphRepList.cswap( myList, NULL ) );
//   return myList;
//}


// comm_accesses is a map of access:owner for commutative accesess
int WorkDescriptor::getConcurrencyLevel( std::map<WD**, WD*> &comm_accesses ) const
{
   int num_wds = 0;

   // Slicer: return from 1 to N
   if ( _slicer != NULL ) {
      nanos_loop_info_t *loop_info;
      loop_info = ( nanos_loop_info_t* ) _data;
      int64_t _chunk = loop_info->chunk;
      int64_t _lower = loop_info->lower;
      int64_t _upper = loop_info->upper;
      int64_t _step  = loop_info->step;
      int64_t _niters = (((_upper - _lower) / _step ) + 1 );

      if ( _chunk == 0 ) {
         num_wds = 1;
      } else {
         num_wds = (_niters + _chunk - 1) / _chunk;
      }
   }

   // Commutative: return 0 to 1
   else if ( _commutativeOwners != NULL ) {
      WorkDescriptorPtrList::const_iterator owner_it;
      // Check first that all the WD'a accesses can be acquired
      for ( owner_it = _commutativeOwners->begin();
            owner_it != _commutativeOwners->end();
            ++owner_it ) {
         // WD** that contains the parent's commutative access
         WD **owner_ptr = *owner_it;
         // WD* owner of the actual access
         WD *owner = *owner_ptr;

         // If the access has an owner, update the local structure
         if ( owner && owner != comm_accesses[owner_ptr] ) {
            comm_accesses[owner_ptr] = owner;
         }

         // We stop looking if the access is reserved by other WD
         if ( comm_accesses[owner_ptr] != NULL && comm_accesses[owner_ptr] != (WD*) this ) {
            break;
         }
      }

      // All the WD's accessed can be acquired, register them into comm_accesses
      if ( owner_it == _commutativeOwners->end() ) {
         for ( owner_it = _commutativeOwners->begin();
               owner_it != _commutativeOwners->end();
               ++owner_it ) {
            WD **owner_ptr = *owner_it;
            comm_accesses[owner_ptr] = (WD*) this;
         }
         num_wds = 1;
      }
   }

   // Normal non-tied task: return 1
   else if ( _tiedTo == NULL ) {
      num_wds = 1;
   }

   return num_wds;
}

void WorkDescriptor::addPresubmittedWDs( unsigned int numWDs, WD **wds ) {
   bool delay = false;
   if ( wds[0]->_parent == this ) {
      /* Im the parent of these WDs */
      delay = !_reachedTaskwait;
   } else if ( _parent != NULL && wds[0]->_parent == _parent ) {
      /* Im a sibling */
      delay = !(_parent->_reachedTaskwait);

   }
   if ( delay ) {
      if ( _submittedWDs == NULL ) {
         _submittedWDs = NEW std::vector< WD * >( &wds[0], &wds[numWDs] );
      } else {
         std::size_t orig_size = _submittedWDs->size();
         _submittedWDs->resize( orig_size + numWDs );
         for ( std::size_t idx = orig_size; idx < orig_size + numWDs; idx += 1 ) {
            (*_submittedWDs)[idx] = wds[idx - orig_size];
         }
      }
   } else {
      Scheduler::_submit( wds, numWDs );
   }
}
