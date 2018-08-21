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

#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "config.hpp"
#include "basethread.hpp"
#include <iostream>
#ifdef CLUSTER_DEV
#include "clusterthread_decl.hpp"
#endif

using namespace nanos;
using namespace nanos::ext;

bool SMPProcessor::_useUserThreads = true;
size_t SMPProcessor::_threadsStackSize = 0;
System::CachePolicyType SMPProcessor::_cachePolicy = System::DEFAULT;
size_t SMPProcessor::_cacheDefaultSize = 1048580;

SMPProcessor::SMPProcessor( int bindingId, const CpuSet& bindingList,
      memory_space_id_t memId, bool active, unsigned int numaNode, unsigned int socket ) :
   PE( &getSMPDevice(), memId, 0 /* always local node */, numaNode, true, socket, true ),
   _bindingId( bindingId ), _bindingList( bindingList ),
   _reserved( false ), _active( active ), _futureThreads( 0 ) {}

void SMPProcessor::prepareConfig ( Config &config )
{
   config.registerConfigOption( "user-threads", NEW Config::FlagOption( _useUserThreads, false), "Disable User Level Threads" );
   config.registerArgOption( "user-threads", "disable-ut" );

   config.registerConfigOption ( "thread-stack-size", NEW Config::SizeVar( _threadsStackSize ), "Defines thread stack size" );
   config.registerArgOption( "thread-stack-size", "thread-stack-size" );
   config.registerEnvOption( "thread-stack-size", "OMP_STACKSIZE" );
}

WorkDescriptor & SMPProcessor::getWorkerWD () const
{
   SMPDD  *dd = NEW SMPDD( ( SMPDD::work_fct )Scheduler::workerLoop );
   DeviceData **dd_ptr = NEW DeviceData*[1];
   dd_ptr[0] = (DeviceData*)dd;

   WD * wd = NEW WD( 1, dd_ptr, 0, 1, 0, 0, NULL, NULL, "SMP Worker" );

   return *wd;
}

WorkDescriptor & SMPProcessor::getMultiWorkerWD () const
{
#ifdef CLUSTER_DEV
   SMPDD * dd = NEW SMPDD( ( SMPDD::work_fct )ClusterThread::workerClusterLoop );
#else
   SMPDD * dd = NEW SMPDD( ( SMPDD::work_fct )NULL); //this is an error if we are not runnig with cluster
#endif
   WD *wd = NEW WD( dd, 0, 1, 0, 0, NULL, NULL, "SMP MultiWorker" );
   wd->_mcontrol.preInit();

   return *wd;
}

WorkDescriptor & SMPProcessor::getMasterWD () const
{
   SMPDD  *dd = NEW SMPDD();
   DeviceData **dd_ptr = NEW DeviceData*[1];
   dd_ptr[0] = (DeviceData*)dd;

   WD * wd = NEW WD( 1, dd_ptr, 0, 1, 0, 0, NULL, NULL, "SMP Main" );

   return *wd;
}

BaseThread &SMPProcessor::createThread ( WorkDescriptor &helper, SMPMultiThread *parent )
{
   ensure( helper.canRunIn( getSMPDevice() ),"Incompatible worker thread" );
   SMPThread &th = *NEW SMPThread( helper, this, this );
   th.stackSize( _threadsStackSize ).useUserThreads( _useUserThreads );

   return th;
}

BaseThread &SMPProcessor::createMultiThread ( WorkDescriptor &helper, unsigned int numPEs, PE **repPEs )
{
   ensure( helper.canRunIn( getSMPDevice() ),"Incompatible worker thread" );
   SMPThread &th = *NEW SMPMultiThread( helper, this, numPEs, repPEs );
   th.stackSize(_threadsStackSize).useUserThreads(_useUserThreads);

   return th;
}

SMPThread &SMPProcessor::associateThisThread( bool untieMain ) {

   WD & master = getMasterWD();
   WD & worker = getWorkerWD();

   NANOS_INSTRUMENT (sys.getInstrumentation()->raiseOpenPtPEvent ( NANOS_WD_DOMAIN, (nanos_event_id_t) master.getId(), 0, 0 ); )
   NANOS_INSTRUMENT (InstrumentationContextData *icd = master.getInstrumentationContextData() );
   NANOS_INSTRUMENT (icd->setStartingWD(true) );

   SMPThread &thread = (SMPThread &)createThread( worker );

   thread.initMain();
   thread.setMainThread();
   thread.associate( &master );
   worker._mcontrol.preInit();

   getThreads().push_back( &thread );

   if ( !untieMain ) {
      master.tieTo(thread);
   }

   return thread;
}

void SMPProcessor::setNumFutureThreads( unsigned int nthreads ) {
   _futureThreads = nthreads;
}

unsigned int SMPProcessor::getNumFutureThreads() const {
   return _futureThreads;
}
