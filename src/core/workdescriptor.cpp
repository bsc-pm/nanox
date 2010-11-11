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
#include "dependableobjectwd.hpp"
#include "system.hpp"

using namespace nanos;

void WorkDescriptor::init (bool isUserLevelThread, WorkDescriptor *previous)
{
   BaseThread *myThd = getMyThreadSafe();
   ProcessingElement *pe = getPe();

   setPrevious( myThd->getCurrentWD() );

   if (pe == NULL) {
      pe = myThd->runningOn();
      setPe( pe );
   }

   //std::cerr << "thd: " << myThread->getId() << " -- Starting wd " << this << ":" << getId() << " pe: " << pe << " is ULT? " << isUserLevelThread << " previous " << previous << " current " << &myThread->getThreadWD() << std::endl;

   /* Initializing instrumentation context */
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdCreate( this ) );

   _activeDevice->lazyInit(*this,isUserLevelThread,previous);
   
   if ( getNumCopies() > 0 )
   {
      pe->copyDataIn( *this );
   }

   setReady();
}

void WorkDescriptor::start()
{
   ProcessingElement *pe = myThread->runningOn();

   if ( getNumCopies() > 0 )
      pe->waitInputs( *this );
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

   ensure( dd,"Did not find requested device in activation" );
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
   if ( started() && !pe.supportsUserLevelThreads() ) return false;
   return canRunIn( pe.getDeviceType() );
}

void WorkDescriptor::submit( void )
{
   Scheduler::submit( *this );
} 

void WorkDescriptor::done ()
{
   //ProcessingElement *pe = myThread->runningOn();
   //ProcessingElement *pe = myThread->getThreadWD().getPe();
   ProcessingElement *pe = getPe();

   if (pe == NULL)
      pe = getMyThreadSafe()->runningOn();

   if ( pe->hasSeparatedMemorySpace() )
   {
      //std::cerr <<  "has separate MS ; node " << sys.getNetwork()->getNodeNum() << " wd " << this << " pe is " << pe << std::endl;
     pe->copyDataOut( *this );
   }

   // FIX-ME: We are waiting for the children tasks to avoid to keep alive only part of the parent
   waitCompletion();
   this->getParent()->workFinished( *this );

   //std::cerr << "wg done wd " << getId() << std::endl;
   WorkGroup::done();

}

void WorkDescriptor::prepareCopies()
{
   for (unsigned int i = 0; i < _numCopies; i++ ) {
      if ( _copies[i].isPrivate() )
         _copies[i].setAddress( ( (uint64_t)_copies[i].getAddress() - (unsigned long)_data ) );
   }
}

