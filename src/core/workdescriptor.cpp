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

   _executionTime = ( _numDevices == 1 ? 0.0 : OS::getMonotonicTimeUs() );

   if ( getNumCopies() > 0 ) {
      pe->copyDataIn( *this );
      if ( _translateArgs != NULL ) {
         _translateArgs( _data, this );
      }
   }
   setStart();
}

void WorkDescriptor::start(ULTFlag isUserLevelThread, WorkDescriptor *previous)
{
   ensure ( _state == START , "Trying to start a wd twice or trying to start an uninitialized wd");

   ensure ( _activeDevice != NULL, "WD's _activeDevice is not set. If you are using 'implements' feature, please use versioning scheduler." );

   _activeDevice->lazyInit(*this,isUserLevelThread,previous);
   
   ProcessingElement *pe = myThread->runningOn();

   if ( getNumCopies() > 0 )
      pe->waitInputs( *this );

   if ( _tie ) tieTo(*myThread);

   sys.getPMInterface().wdStarted( *this );
   setReady();
}

void WorkDescriptor::prepareDevice ()
{
   // Do nothing if there is already an active device
   if ( _activeDevice ) return;

   if ( _numDevices == 1 ) {
      _activeDevice = _devices[0];
      _activeDeviceIdx = 0;
      return;
   }

   // Choose between the supported devices
   message("No active device --> selecting one");
   _activeDevice = _devices[_numDevices-1];
   _activeDeviceIdx = _numDevices-1;
}

DeviceData & WorkDescriptor::activateDevice ( const Device &device )
{
   if ( _activeDevice ) {
      ensure( _activeDevice->isCompatible( device ),"Bogus double device activation" );
      return *_activeDevice;
   }
   unsigned i = _numDevices;
   for ( i = 0; i < _numDevices; i++ ) {
      if ( _devices[i]->isCompatible( device ) ) {
         _activeDevice = _devices[i];
         _activeDeviceIdx = i;
         break;
      }
   }

   ensure( i < _numDevices, "Did not find requested device in activation" );
   return *_activeDevice;
}

DeviceData & WorkDescriptor::activateDevice ( unsigned int deviceIdx )
{
   ensure( _numDevices > deviceIdx, "The requested device does not exist" );

   _activeDevice = _devices[deviceIdx];
   _activeDeviceIdx = deviceIdx;

   return *_activeDevice;
}

bool WorkDescriptor::canRunIn( const Device &device ) const
{
   if ( _activeDevice ) return _activeDevice->isCompatible( device );

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

   _executionTime = ( _numDevices == 1 ? 0.0 : OS::getMonotonicTimeUs() - _executionTime );
}

void WorkDescriptor::done ()
{
   sys.getPMInterface().wdFinished( *this );
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

