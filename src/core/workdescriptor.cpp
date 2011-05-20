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
   ProcessingElement *pe = myThread->runningOn();

   /* Initializing instrumentation context */
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdCreate( this ) );

   _executionTime = OS::getMonotonicTime();
   if ( getNumCopies() > 0 ) {
      pe->copyDataIn( *this );
      if ( _translateArgs != NULL ) {
         _translateArgs( _data, this );
      }
   }
}

void WorkDescriptor::start(ULTFlag isUserLevelThread, WorkDescriptor *previous)
{
   _activeDevice->lazyInit(*this,isUserLevelThread,previous);
   
   ProcessingElement *pe = myThread->runningOn();

   if ( getNumCopies() > 0 )
      pe->waitInputs( *this );

   if ( _tie ) tieTo(*myThread);

   setReady();
}

void WorkDescriptor::prepareDevice ()
{
   // Do nothing if there is already an active device
   if ( _activeDevice ) return;

   if ( _numDevices == 1 ) {
      _activeDevice = _devices[0];
      return;
   }

   // Choose between the supported devices
   message("No active device --> selecting one");
   _activeDevice = _devices[_numDevices-1];
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
   Scheduler::submit(*this);
} 

void WorkDescriptor::finish ()
{
   ProcessingElement *pe = myThread->runningOn();
   waitCompletionAndSignalers();
   if ( getNumCopies() > 0 )
     pe->copyDataOut( *this );

   _executionTime = OS::getMonotonicTime() - _executionTime;
}

void WorkDescriptor::done ()
{
   this->getParent()->workFinished( *this );
   WorkGroup::done();
}

void WorkDescriptor::prepareCopies()
{
   for (unsigned int i = 0; i < _numCopies; i++ ) {
      _copiesSize += _copies[i].getSize();

      if ( _copies[i].isPrivate() )
         _copies[i].setAddress( ( (uint64_t)_copies[i].getAddress() - (unsigned long)_data ) );
   }

   _paramsSize = _copiesSize;
}

