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

#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "config.hpp"
#include <iostream>

using namespace nanos;
using namespace nanos::ext;

bool SMPProcessor::_useUserThreads = true;
size_t SMPProcessor::_threadsStackSize = 0;
System::CachePolicyType SMPProcessor::_cachePolicy = System::DEFAULT;
size_t SMPProcessor::_cacheDefaultSize = 1048580;

SMPProcessor::SMPProcessor( int bindingId, memory_space_id_t memId, bool active, unsigned int numaNode, unsigned int socket ) :
   PE( &SMP, NULL, memId, 0 /* always local node */, numaNode, true, socket, true ),
   _bindingId( bindingId ), _reserved(false), _active( active ) {}

void SMPProcessor::prepareConfig ( Config &config )
{
   config.registerConfigOption( "user-threads", NEW Config::FlagOption( _useUserThreads, false), "Disable use of user threads to implement workdescriptor" );
   config.registerArgOption( "user-threads", "disable-ut" );

   config.registerConfigOption ( "pthreads-stack-size", NEW Config::SizeVar( _threadsStackSize ), "Defines pthreads stack size" );
   config.registerArgOption( "pthreads-stack-size", "pthreads-stack-size" );
}

WorkDescriptor & SMPProcessor::getWorkerWD () const
{
   SMPDD  *dd = NEW SMPDD( ( SMPDD::work_fct )Scheduler::workerLoop );
   DeviceData **dd_ptr = NEW DeviceData*[1];
   dd_ptr[0] = (DeviceData*)dd;

   WD *wd = NEW WD( 1, dd_ptr );

   return *wd;
}

WorkDescriptor & SMPProcessor::getMultiWorkerWD () const
{
#ifdef CLUSTER_DEV
   SMPDD * dd = NEW SMPDD( ( SMPDD::work_fct )Scheduler::workerClusterLoop );
#else
   SMPDD * dd = NEW SMPDD( ( SMPDD::work_fct )NULL); //this is an error if we are not runnig with cluster
#endif
   WD *wd = NEW WD( dd );
   return *wd;
}

WorkDescriptor & SMPProcessor::getMasterWD () const
{
   SMPDD  *dd = NEW SMPDD();
   DeviceData **dd_ptr = NEW DeviceData*[1];
   dd_ptr[0] = (DeviceData*)dd;

   WD * wd = NEW WD( 1, dd_ptr );

   return *wd;
}

BaseThread &SMPProcessor::createThread ( WorkDescriptor &helper, SMPMultiThread *parent )
{
   ensure( helper.canRunIn( SMP ),"Incompatible worker thread" );
   SMPThread &th = *NEW SMPThread( helper, this, this );
   th.stackSize( _threadsStackSize ).useUserThreads( _useUserThreads );

   return th;
}

BaseThread &SMPProcessor::createMultiThread ( WorkDescriptor &helper, unsigned int numPEs, PE **repPEs )
{
   ensure( helper.canRunIn( SMP ),"Incompatible worker thread" );
   SMPThread &th = *NEW SMPMultiThread( helper, this, numPEs, repPEs );
   th.stackSize(_threadsStackSize).useUserThreads(_useUserThreads);

   return th;
}

SMPThread &SMPProcessor::associateThisThread( bool untieMain ) {
   WD & worker = getMasterWD();
   NANOS_INSTRUMENT (sys.getInstrumentation()->raiseOpenPtPEvent ( NANOS_WD_DOMAIN, (nanos_event_id_t) worker.getId(), 0, 0 ); )
   NANOS_INSTRUMENT (InstrumentationContextData *icd = worker.getInstrumentationContextData() );
   NANOS_INSTRUMENT (icd->setStartingWD(true) );
   
   SMPThread &thread = (SMPThread &)createThread( worker );

   thread.setMainThread();
   thread.associate();

   getThreads().push_back( &thread );

   if ( !untieMain ) {
      worker.tieTo(thread);
   }

   return thread;
}

