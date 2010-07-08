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


WorkDescriptor::WorkDescriptor ( int ndevices, DeviceData **devs, size_t data_size, void *wdata,
                                 size_t numCopies, CopyData *copies ) :
              WorkGroup(), _data_size ( data_size ), _data ( wdata ), _wdData ( 0 ), _tie ( false ), _tiedTo ( 0 ),
             _state( INIT ), _syncCond( NULL ),  _parent ( NULL ), _myQueue ( NULL ), _depth ( 0 ),
             _numDevices ( ndevices ), _devices ( devs ), _activeDevice ( ndevices == 1 ? devs[0] : 0 ),
             _numCopies( numCopies ), _copies( copies ), _doSubmit(), _doWait(),
             _depsDomain(), _instrumentorContext(NULL)
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   if ( sys.getInstrumentor()->useStackedBursts() ) _instrumentorContext = new InstrumentationContextStackedBursts();
   else _instrumentorContext = new InstrumentationContext();
#endif
}

WorkDescriptor::WorkDescriptor ( DeviceData *device, size_t data_size, void *wdata, size_t numCopies, CopyData *copies ) :
              WorkGroup(), _data_size ( data_size ), _data ( wdata ), _wdData ( 0 ), _tie ( false ), _tiedTo ( 0 ),
              _state( INIT ), _syncCond( NULL ), _parent ( NULL ), _myQueue ( NULL ), _depth ( 0 ),
              _numDevices ( 1 ), _devices ( &_activeDevice ), _activeDevice ( device ),
              _numCopies( numCopies ), _copies( copies ), _doSubmit(), _doWait(),
              _depsDomain(), _instrumentorContext(NULL)
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   if ( sys.getInstrumentor()->useStackedBursts() ) _instrumentorContext = new InstrumentationContextStackedBursts();
   else _instrumentorContext = new InstrumentationContext();
#endif
}

WorkDescriptor::WorkDescriptor ( const WorkDescriptor &wd, DeviceData **devs, CopyData * copies, void *data ) :
                    WorkGroup( wd ), _data_size( wd._data_size ), _data ( data ), _wdData ( NULL ),
                    _tie ( wd._tie ), _tiedTo ( wd._tiedTo ), _state ( INIT ), _syncCond( NULL ), _parent ( wd._parent ),
                    _myQueue ( NULL ), _depth ( wd._depth ), _numDevices ( wd._numDevices ),
                    _devices ( devs ), _activeDevice ( wd._numDevices == 1 ? devs[0] : NULL ),
                    _numCopies( wd._numCopies ), _copies( wd._numCopies == 0 ? NULL : copies ),
                    _doSubmit(), _doWait(), _depsDomain(), _instrumentorContext( NULL )
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
            if ( sys.getInstrumentor()->useStackedBursts() ) _instrumentorContext = new InstrumentationContextStackedBursts( wd._instrumentorContext );  
            else _instrumentorContext = new InstrumentationContext( wd._instrumentorContext );   
#endif
}

void WorkDescriptor::start (bool isUserLevelThread, WorkDescriptor *previous)
{
   ProcessingElement *pe = myThread->runningOn();

   /* Initializing instrumentor context */
   NANOS_INSTRUMENT( sys.getInstrumentor()->wdCreate( this ) );

   _activeDevice->lazyInit(*this,isUserLevelThread,previous);
   
   if ( getNumCopies() > 0 && pe->hasSeparatedMemorySpace() )
      pe->copyDataIn( *this );

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
   return canRunIn( pe.getDeviceType() );
}

void WorkDescriptor::submit( void )
{
   Scheduler::submit(*this);
} 

void WorkDescriptor::done ()
{
   ProcessingElement *pe = myThread->runningOn();
   if ( pe->hasSeparatedMemorySpace() )
     pe->copyDataOut( *this );

   // FIX-ME: We are waiting for the children tasks to avoid to keep alive only part of the parent
   waitCompletion();
   this->getParent()->workFinished( *this );
   WorkGroup::done();
}

void WorkDescriptor::prepareCopies()
{
   for (unsigned int i = 0; i < _numCopies; i++ ) {
      if ( _copies[i].isPrivate() )
         _copies[i].setAddress( ( (uint64_t)_copies[i].getAddress() - (unsigned long)_data ) );
   }
}

