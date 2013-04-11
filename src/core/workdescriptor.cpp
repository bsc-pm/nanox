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


using namespace nanos;

void WorkDescriptor::init ()
{
   if ( _state != INIT ) return;

   ProcessingElement *pe = myThread->runningOn();

   /* Initializing instrumentation context */
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdCreate( this ) );
  
   //message("init wd " << getId() );
   //if ( getNewDirectory() == NULL )
   //   initNewDirectory();
   //getNewDirectory()->setParent( ( getParent() != NULL ) ? getParent()->getNewDirectory() : NULL );   

   _executionTime = ( _numDevices == 1 ? 0.0 : OS::getMonotonicTimeUs() );

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
      
      _notifyThread = myThread;
      pe->copyDataIn( *this );
      //this->notifyCopy();

      if ( _translateArgs != NULL ) {
         _translateArgs( _data, this );
      }
   }
   setStart();
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
   setStart();
}


void WorkDescriptor::notifyCopy()
{
   if ( _notifyCopy != NULL ) {
      //std::cerr << " WD " << getId() << " GONNA CALL THIS SHIT IF POSSIBLE: " << (void *)_notifyCopy << " ARG IS THD "<< _notifyThread->getId() << std::endl;
      _notifyCopy( *this, *_notifyThread );
   }
}

// That function must be called from the thread it will execute it. This is important
// from the point of view of tiedness and the device activation. Both operation will
// involves current thread / pe
void WorkDescriptor::start (ULTFlag isUserLevelThread, WorkDescriptor *previous)
{
   ensure ( _state == START , "Trying to start a wd twice or trying to start an uninitialized wd");

   ProcessingElement *pe = myThread->runningOn();


   //if ( getNumCopies() > 0 ) {
   //   pe->waitInputs( *this );
   //}

   //if ( getNumCopies() > 0 ) {
   //   if ( _ccontrol.dataIsReady() ) {
   //      //message("Data is Ready!");
   //   }
   //}

   // If there are no active device, choose a compatible one
   if ( _activeDeviceIdx == _numDevices ) activateDevice ( *(pe->getDeviceType()) );

   // Initializing devices
   _devices[_activeDeviceIdx]->lazyInit( *this, isUserLevelThread, previous );

   if ( sys.getNetwork()->getNodeNum() == 0 )fprintf(stderr, "[%d] ============= WAIT INPUTS ========> wd %p (%d)\n", sys.getNetwork()->getNodeNum(), this, getId() );
   // Waiting for copies
   if ( getNumCopies() > 0 ) pe->waitInputs( *this );
   if ( sys.getNetwork()->getNodeNum() == 0 )fprintf(stderr, "[%d] ============= WAIT INPUTS DONE ========> wd %p (%d)\n", sys.getNetwork()->getNodeNum(), this, getId() );

   // Tie WD to current thread
   if ( _tie ) tieTo( *myThread );

   // Call Programming Model interface .started() method.
   sys.getPMInterface().wdStarted( *this );

   // Setting state to ready
   setReady();
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
   bool result;
   if ( started() && !pe.supportsUserLevelThreads() ) return false;

   if ( pe.getDeviceType() == NULL )  result = canRunIn( *pe.getSubDeviceType() );
   else result = canRunIn( *pe.getDeviceType() ) ;

   return result;
   //return ( canRunIn( pe.getDeviceType() )  || ( pe.getSubDeviceType() != NULL && canRunIn( *pe.getSubDeviceType() ) ));
}

void WorkDescriptor::submit( void )
{
   //if ( !sys.usingNewCache() ) {
   //if ( getNewDirectory() == NULL )
   //   initNewDirectory();
   //   getNewDirectory()->setParent( ( getParent() != NULL ) ? getParent()->getNewDirectory() : NULL );   
   //}
   //_ccontrol.preInit();
   _mcontrol.preInit();
   Scheduler::submit( *this );
} 

void WorkDescriptor::finish ()
{
   //ProcessingElement *pe = myThread->runningOn();
   waitCompletion( true );
   if ( getNumCopies() > 0 )
      _mcontrol.copyDataOut();
   

   _executionTime = ( _numDevices == 1 ? 0.0 : OS::getMonotonicTimeUs() - _executionTime );
}

void WorkDescriptor::done ()
{
   releaseCommutativeAccesses(); 

   sys.getPMInterface().wdFinished( *this );
   this->getParent()->workFinished( *this );

   this->wgdone();
   WorkGroup::done();

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

   //if ( _myGraphRepList == NULL ) {
   //   _myGraphRepList = sys.getGraphRepList();
   //   if ( predecessorWd != NULL ) {
   //      _myGraphRepList.value()->push_back( predecessorWd->getGE() );
   //      predecessorWd->listed();
   //   }
   //}
   //_myGraphRepList.value()->push_back( getGE() );
   //if (predecessorWd != NULL) predecessorWd->listed();

   //message("wd " << getId() << " getting directory from previous wd " << predecessorWd->getId() );
   //if ( !sys.usingNewCache() ) {
   //   if ( getNewDirectory() == NULL )
   //      initNewDirectory();
   //   //std::cerr << "wd " << (unsigned int)  getId() << " getting directory from " << (unsigned int)predecessorWd->getId() << std::endl;
   //   _newDirectory->merge( *predecessorWd->getNewDirectory() );
   //} else {
      //_ccontrol.getInfoFromPredecessor( predecessorWd->_ccontrol ); 
      _mcontrol.getInfoFromPredecessor( predecessorWd->_mcontrol ); 
   //}
}

#if 0
void WorkDescriptor::initMyGraphRepListNoPred( )
{
   _myGraphRepList = sys.getGraphRepList();
   _myGraphRepList.value()->push_back( this->getParent()->getGE() );
   _myGraphRepList.value()->push_back( getGE() );
}
void WorkDescriptor::setMyGraphRepList( std::list<GraphEntry *> *myList )
{
   _myGraphRepList = myList;
}
std::list<GraphEntry *> *WorkDescriptor::getMyGraphRepList(  )
{
   std::list<GraphEntry *> *myList;
   do {
      myList = _myGraphRepList.value();
   }
   while ( ! _myGraphRepList.cswap( myList, NULL ) );
   return myList;
}
#endif

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

void WorkDescriptor::listed()
{
   _listed = true;
}

void WorkDescriptor::printCopies()
{
      CopyData *copies = getCopies();
      for ( unsigned int i = 0; i < getNumCopies(); i++ ) {
         CopyData & cd = copies[i];
         if ( !cd.isPrivate() ) {
            //message("WD: " << getId() << " DA["<< i << "]: addr is " << (void *) cd.getDataAccess()->address );
            //DataAccess d( cd.getDataAccess().address, cd.getDataAccess().flags.input ,cd.getDataAccess().flags.output, cd.getDataAccess().flags.can_rename,
            //   cd.getDataAccess().flags.commutative, cd.getDataAccess().dimension_count, cd.getDataAccess().dimensions);
            //  Region reg = NewRegionDirectory::build_region( d );
         }
      }

}
void WorkDescriptor::workFinished(WorkDescriptor &wd)
{
   //if ( !sys.usingNewCache() ) {
   //   if ( _newDirectory == NULL ) initNewDirectory();
   //   //if (sys.getNetwork()->getNodeNum() > 0 ) std::cerr << "wd " << (unsigned int) wd.getId() <<":" <<(void *) wd.getNewDirectory() << " merging directory into parent " << (unsigned int) this->getId() <<":"<<(void *) _newDirectory << " num successors " << (int) ( wd._doSubmit != NULL ? wd._doSubmit->getSuccessors().size() : -1 ) << std::endl;
   //   NANOS_INSTRUMENT( InstrumentState inst3(NANOS_POST_OUTLINE_WORK3); );
   //   _newDirectory->mergeOutput( *(wd.getNewDirectory()) );
   //   NANOS_INSTRUMENT( inst3.close(); );
   //   //if (sys.getNetwork()->getNodeNum() > 0) {
   //   //   _newDirectory->consolidate( false );
   //   //}
   //}

   if ( wd._doSubmit != NULL )
      wd._doSubmit->finished();
   //if (sys.getNetwork()->getNodeNum()==0){ message("a child, " << wd.getId() << " has finished, im " << getId() ); }
}
void WorkDescriptor::setNotifyCopyFunc( void (*func)(WD &, BaseThread const&) ) {
   _notifyCopy = func;
}

unsigned int WorkDescriptor::getNumReaders() {
   unsigned int result = 0;
   //CopyData *copies = getCopies();
   //for ( unsigned int i = 0; i < getNumCopies(); i++ ) {
   //   CopyData & cd = copies[i];
   //   if ( !cd.isPrivate() && cd.isOutput() ) {
   //      result += (getParent()->_depsDomain)->getNumReaders( _ccontrol._cacheCopies[i]._region );
   //   }
   //}
   //std::cerr << "getNumReaders: succ from dosubmit "<< _doSubmit->getSuccessors().size() << std::endl;
   result = _doSubmit->getSuccessors().size();
   return result;
}

unsigned int WorkDescriptor::getNumAllReaders() {
   //unsigned int result = 0;
   //CopyData *copies = getCopies();
   //for ( unsigned int i = 0; i < getNumCopies(); i++ ) {
   //   CopyData & cd = copies[i];
   //   if ( !cd.isPrivate() && cd.isOutput() ) {
   //   }
   //}
   //return result;
    //return (getParent()->_depsDomain)->getNumAllReaders(  );
  return 0;
}

void WorkDescriptor::initCommutativeAccesses( WorkDescriptor &wd, size_t numDeps, DataAccess* deps )
{
   size_t numCommutative = 0;

   for ( size_t i = 0; i < numDeps; i++ )
      if ( deps[i].isCommutative() )
         ++numCommutative;

   if ( numCommutative == 0 )
      return;

   wd._commutativeOwners.reserve(numCommutative);

   for ( size_t i = 0; i < numDeps; i++ ) {
      if ( !deps[i].isCommutative() )
         continue;

      // Lookup owner in map in parent WD
      CommutativeOwnerMap::iterator iter = _commutativeOwnerMap.find( deps[i].getDepAddress() );

      if ( iter != _commutativeOwnerMap.end() ) {
         // Already in map => insert into owner list in child WD
         wd._commutativeOwners.push_back( iter->second.get() );
      }
      else {
         // Not in map => allocate new owner pointer container and insert
         std::pair<CommutativeOwnerMap::iterator, bool> ret =
               _commutativeOwnerMap.insert( std::make_pair( deps[i].getDepAddress(),
                                                            TR1::shared_ptr<WorkDescriptor *>( NEW WorkDescriptor *(NULL) ) ) );

         // Insert into owner list in child WD
         wd._commutativeOwners.push_back( ret.first->second.get() );
      }
   }
}

bool WorkDescriptor::tryAcquireCommutativeAccesses()
{
   const size_t n = _commutativeOwners.size();
   for ( size_t i = 0; i < n; i++ ) {

      WorkDescriptor *owner = *_commutativeOwners[i];

      if ( owner == this )
         continue;

      if ( owner == NULL &&
           nanos::compareAndSwap( (void **) _commutativeOwners[i], (void *) NULL, (void *) this ) )
         continue;

      // Failed to obtain exclusive access to all accesses, release the obtained ones

      for ( ; i > 0; i-- )
         *_commutativeOwners[i-1] = NULL;

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

