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

#include <string.h>
#include "processingelement.hpp"
#include "basethread.hpp"
#include "debug.hpp"
#include "schedule.hpp"
#include "copydata.hpp"
#include "system.hpp"
#include "instrumentation.hpp"

using namespace nanos;

void ProcessingElement::copyDataIn( WorkDescriptor &work )
{
   //work._ccontrol.copyDataIn( NULL );
   work._mcontrol.copyDataIn();
}

void ProcessingElement::copyDataOut( WorkDescriptor &work )
{
}

void ProcessingElement::waitInputs( WorkDescriptor &work )
{
   BaseThread * thread = getMyThreadSafe();
   //while ( !work._ccontrol.dataIsReady() ) { 
   while ( !work._mcontrol.isDataReady( work ) ) { 
      thread->idle();
      thread->getTeam()->getSchedulePolicy().atSupport( thread ); 
   }
   //if( sys.getNetwork()->getNodeNum() == 0 && work._mcontrol.getMaxAffinityScore() > 0) {
   //   std::cerr << "WD " << work.getId() << " affinity score " << work._mcontrol.getAffinityScore() << " (max "<< work._mcontrol.getMaxAffinityScore() <<") and has transferred " << work._mcontrol.getAmountOfTransferredData() << " total wd data " << work._mcontrol.getTotalAmountOfData() << " dev: ";
   //   if ( ( work._mcontrol.getTotalAmountOfData() - work._mcontrol.getAmountOfTransferredData() ) == work._mcontrol.getAffinityScore() ) {
   //      std::cerr << " as expected " << work._mcontrol.getAffinityScore() - ( work._mcontrol.getTotalAmountOfData() - work._mcontrol.getAmountOfTransferredData() ) << std::endl;
   //   } else if ( ( work._mcontrol.getTotalAmountOfData() - work._mcontrol.getAmountOfTransferredData() ) > work._mcontrol.getAffinityScore() ) {
   //      std::cerr << " less than expected data transferred " <<  ( work._mcontrol.getTotalAmountOfData() - work._mcontrol.getAmountOfTransferredData() ) - work._mcontrol.getAffinityScore() << std::endl;
   //   } else if ( ( work._mcontrol.getTotalAmountOfData() - work._mcontrol.getAmountOfTransferredData() ) < work._mcontrol.getAffinityScore() ) {
   //      std::cerr << " more than expected data transferred " << work._mcontrol.getAffinityScore() - ( work._mcontrol.getTotalAmountOfData() - work._mcontrol.getAmountOfTransferredData() ) << std::endl;
   //   }
   //}
}

bool ProcessingElement::testInputs( WorkDescriptor &work ) {
   bool result = work._mcontrol.isDataReady( work );
   //if( sys.getNetwork()->getNodeNum() == 0 && work._mcontrol.getMaxAffinityScore() > 0) {
   //   std::cerr << "WD " << work.getId() << " affinity score " << work._mcontrol.getAffinityScore() << " (max "<< work._mcontrol.getMaxAffinityScore() <<") and has transferred " << work._mcontrol.getAmountOfTransferredData() << " total wd data " << work._mcontrol.getTotalAmountOfData() << " dev: ";
   //   if ( ( work._mcontrol.getTotalAmountOfData() - work._mcontrol.getAmountOfTransferredData() ) == work._mcontrol.getAffinityScore() ) {
   //      std::cerr << " as expected " << work._mcontrol.getAffinityScore() - ( work._mcontrol.getTotalAmountOfData() - work._mcontrol.getAmountOfTransferredData() ) << std::endl;
   //   } else if ( ( work._mcontrol.getTotalAmountOfData() - work._mcontrol.getAmountOfTransferredData() ) > work._mcontrol.getAffinityScore() ) {
   //      std::cerr << " less than expected data transferred " <<  ( work._mcontrol.getTotalAmountOfData() - work._mcontrol.getAmountOfTransferredData() ) - work._mcontrol.getAffinityScore() << std::endl;
   //   } else if ( ( work._mcontrol.getTotalAmountOfData() - work._mcontrol.getAmountOfTransferredData() ) < work._mcontrol.getAffinityScore() ) {
   //      std::cerr << " more than expected data transferred " << work._mcontrol.getAffinityScore() - ( work._mcontrol.getTotalAmountOfData() - work._mcontrol.getAmountOfTransferredData() ) << std::endl;
   //   }
   //}
   return result;
}

BaseThread& ProcessingElement::startWorker ( ext::SMPMultiThread *parent )
{
   WD & worker = getWorkerWD();

   NANOS_INSTRUMENT (sys.getInstrumentation()->raiseOpenPtPEvent ( NANOS_WD_DOMAIN, (nanos_event_id_t) worker.getId(), 0, 0 ); )
   NANOS_INSTRUMENT (InstrumentationContextData *icd = worker.getInstrumentationContextData() );
   NANOS_INSTRUMENT (icd->setStartingWD(true) );

   return startThread( worker, parent );
}

BaseThread& ProcessingElement::startMultiWorker ( unsigned int numPEs, ProcessingElement **repPEs )
{
   WD & worker = getMultiWorkerWD();

   NANOS_INSTRUMENT (sys.getInstrumentation()->raiseOpenPtPEvent ( NANOS_WD_DOMAIN, (nanos_event_id_t) worker.getId(), 0, 0 ); )
   NANOS_INSTRUMENT (InstrumentationContextData *icd = worker.getInstrumentationContextData() );
   NANOS_INSTRUMENT (icd->setStartingWD(true) );

   return startMultiThread( worker, numPEs, repPEs );
}

BaseThread & ProcessingElement::startThread ( WD &work, ext::SMPMultiThread *parent )
{
   BaseThread &thread = createThread( work, parent );

   thread.start();

   _threads.push_back( &thread );

   return thread;
}

BaseThread & ProcessingElement::startMultiThread ( WD &work, unsigned int numPEs, PE **repPEs )
{
   BaseThread &thread = createMultiThread( work, numPEs, repPEs );

   thread.start();

   _threads.push_back( &thread );

   return thread;
}

BaseThread & ProcessingElement::associateThisThread ( bool untieMain )
{
   WD & worker = getMasterWD();
   NANOS_INSTRUMENT (sys.getInstrumentation()->raiseOpenPtPEvent ( NANOS_WD_DOMAIN, (nanos_event_id_t) worker.getId(), 0, 0 ); )
   NANOS_INSTRUMENT (InstrumentationContextData *icd = worker.getInstrumentationContextData() );
   NANOS_INSTRUMENT (icd->setStartingWD(true) );
   
   BaseThread &thread = createThread( worker );

   thread.associate();

   _threads.push_back( &thread );

   if ( !untieMain ) {
      worker.tieTo(thread);
   }

   return thread;
}

void ProcessingElement::stopAll ()
{
   ThreadList::iterator it;
   BaseThread *thread;

   for ( it = _threads.begin(); it != _threads.end(); it++ ) {
      thread = *it;
      if ( thread->getId() == 0) continue; /* Protection for master thread */
      thread->wakeup();
      thread->stop();
      thread->signal();
      thread->join();
      if ( thread->hasTeam() )
         thread->leaveTeam();
   }
}

Device const *ProcessingElement::getCacheDeviceType() const {
   return NULL;
}

BaseThread* ProcessingElement::getFirstRunningThread()
{
   ThreadList::iterator it;
   for ( it = _threads.begin(); it != _threads.end(); it++ ) {
      if ( (*it)->hasTeam() && !(*it)->isTaggedToSleep() )
         return (*it);
   }
   return NULL;
}

BaseThread* ProcessingElement::getFirstStoppedThread()
{
   ThreadList::iterator it;
   for ( it = _threads.begin(); it != _threads.end(); it++ ) {
      if ( !(*it)->hasTeam() || (*it)->isTaggedToSleep() )
         return (*it);
   }
   return NULL;
}
