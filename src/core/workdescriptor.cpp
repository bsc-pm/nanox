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

void WorkDescriptor::start(ULTFlag isUserLevelThread, WorkDescriptor *previous)
{
   ensure ( _state == START , "Trying to start a wd twice or trying to start an uninitialized wd");

   _activeDevice->lazyInit(*this,isUserLevelThread,previous);
   
   ProcessingElement *pe = myThread->runningOn();

   if ( getNumCopies() > 0 ) {
      pe->waitInputs( *this );
   }

   //if ( getNumCopies() > 0 ) {
   //   if ( _ccontrol.dataIsReady() ) {
   //      //message("Data is Ready!");
   //   }
   //}

   if ( _tie ) tieTo(*myThread);

   sys.getPMInterface().wdStarted( *this );
   setReady();
}

DeviceData * WorkDescriptor::findDeviceData ( const Device &device ) const
{
   for ( unsigned i = 0; i < _numDevices; i++ ) {
      if ( _devices[i]->isCompatible( device ) ) {
         return _devices[i];
      }
   }

   return 0;
}

DeviceData & WorkDescriptor::activateDevice ( const Device &device )
{
   if ( _activeDevice ) {
      ensure( _activeDevice->isCompatible( device ),"Bogus double device activation" );
      return *_activeDevice;
   }

   DD * dd = findDeviceData( device );

//   ensure( dd,"Did not find requested device in activation" );
   _activeDevice = dd;
   return *dd;
}

bool WorkDescriptor::canRunIn( const Device &device ) const
{
   if ( _activeDevice ) return _activeDevice->isCompatible( device );

   return findDeviceData( device ) != NULL;
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
   _mcontrol.preInit();
   Scheduler::submit( *this );
} 

void WorkDescriptor::done ()
{
   waitCompletionAndSignalers( true );

   if ( getNumCopies() > 0 )
      _mcontrol.copyDataOut();
   

   sys.getPMInterface().wdFinished( *this );

               NANOS_INSTRUMENT( InstrumentState inst3(NANOS_POST_OUTLINE_WORK4 ); );
   this->getParent()->workFinished( *this );
               NANOS_INSTRUMENT( inst3.close(); );

   this->wgdone();
   WorkGroup::done();

}

void WorkDescriptor::prepareCopies()
{
   for (unsigned int i = 0; i < _numCopies; i++ ) {
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
   _mcontrol.getInfoFromPredecessor( predecessorWd->_mcontrol ); 
}

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
   if ( wd._doSubmit != NULL )
      wd._doSubmit->finished();
}
void WorkDescriptor::setNotifyCopyFunc( void (*func)(WD &, BaseThread const&) ) {
   _notifyCopy = func;
}
