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

void WorkDescriptor::start (bool isUserLevelThread, WorkDescriptor *previous)
{
   //ProcessingElement *pe = myThread->runningOn();
   //ProcessingElement *pe = myThread->getThreadWD().getPe();
   ProcessingElement *pe = getPe();

   setPrevious( myThread->getCurrentWD() );

   //if (pe != NULL)
   //   setPe(pe);
   //else pe = myThread->runningOn();
   if (pe == NULL) {
      pe = myThread->runningOn();
      //std::cerr << "node " << sys.getNetwork()->getNodeNum() << " wd " << this << " setting pe " << pe << std::endl;
      setPe( pe );
   }

   //std::cerr << "thd: " << myThread->getId() << " -- Starting wd " << this << ":" << getId() << " pe: " << pe << " is ULT? " << isUserLevelThread << " previous " << previous << " current " << &myThread->getThreadWD() << std::endl;
   
   //if (myThread->_counter > 1)
   //{
   //   pe = &pe[_counter - 1];
   //}

   /* Initializing instrumentation context */
   NANOS_INSTRUMENT( sys.getInstrumentation()->wdCreate( this ) );

   _activeDevice->lazyInit(*this,isUserLevelThread,previous);
   
   if ( getNumCopies() > 0 && pe->hasSeparatedMemorySpace() )
   {
      pe->copyDataIn( *this );
   }

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
   if ( started() && !pe.supportsUserLevelThreads() ) return false;
   return canRunIn( pe.getDeviceType() );
}

void WorkDescriptor::submit( void )
{
   Scheduler::submit(*this);
} 

void WorkDescriptor::done ()
{
   //ProcessingElement *pe = myThread->runningOn();
   //ProcessingElement *pe = myThread->getThreadWD().getPe();
   ProcessingElement *pe = getPe();
      //std::cerr <<  "node " << sys.getNetwork()->getNodeNum() << " pe is " << pe << std::endl;

   //if ( sys.getNetwork()->getNodeNum() > 0 )
   //{
   //std::cerr << "WD node: " << sys.getNetwork()->getNodeNum() << " done> ThreadWD addr is " << &myThread->getThreadWD() << std::endl;
   //std::cerr << "WD node: " << sys.getNetwork()->getNodeNum() << " done> pe addr is " << pe << std::endl;
   //std::cerr << "WD node: " << sys.getNetwork()->getNodeNum() << " done> this wd pe addr is " << getPe() << std::endl;
   //}

   if (pe == NULL)
      pe = myThread->runningOn();

   getPrevious()->setNodeFree();

   //if (myThread->_counter > 1)
   //{
   //   pe = &pe[_counter - 1];
   //}

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

   //if (myThread->_counter >= 1)
   //{
   //   myThread->_counter++;
   //   if (myThread->_counter == sys.getNetWork()->getNumNodes())
   //      myThread->_counter = 1;
   //}
}

void WorkDescriptor::prepareCopies()
{
   for (unsigned int i = 0; i < _numCopies; i++ ) {
      if ( _copies[i].isPrivate() )
         _copies[i].setAddress( ( (uint64_t)_copies[i].getAddress() - (unsigned long)_data ) );
   }
}

