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

#include "fpgathread.hpp"
#include "fpgadd.hpp"
#include "fpgamemorytransfer.hpp"
#include "fpgaworker.hpp"

using namespace nanos;
using namespace nanos::ext;

void FPGAThread::initializeDependent()
{
   //initialize device
   ( ( FPGAProcessor * ) myThread->runningOn() )->init();
}


void FPGAThread::runDependent()
{
   verbose( "fpga run dependent" );
   WD &work = getThreadWD();
   setCurrentWD( work );
   SMPDD &dd = ( SMPDD & ) work.activateDevice( getSMPDevice() );
   dd.getWorkFct()( work.getData() );
   ( ( FPGAProcessor * ) myThread->runningOn() )->cleanUp();
}

bool FPGAThread::inlineWorkDependent( WD &wd )
{
   verbose( "fpga nlineWorkDependent" );

   wd.start( WD::IsNotAUserLevelThread );
   FPGAProcessor* fpga = ( FPGAProcessor * ) myThread->runningOn();

   //Instrumentation event to record in which accelerator the task is running
   NANOS_INSTRUMENT (InstrumentBurst i("accelerator#", fpga->getActiveAcc()+1) );
   //set flag to allow new opdate
   fpga->setUpdate(true);
   FPGADD &dd = ( FPGADD & )wd.getActiveDevice();
   ( dd.getWorkFct() )( wd.getData() );

   return true;
}

void FPGAThread::preOutlineWorkDependent ( WD &wd ) {
   wd.preStart(WorkDescriptor::IsNotAUserLevelThread);
}

void FPGAThread::outlineWorkDependent ( WD &wd ) {
   //wd.start( WD::IsNotAUserLevelThread );
   FPGAProcessor* fpga = ( FPGAProcessor * ) myThread->runningOn();

   //Instrumentation event to record in which accelerator the task is running
   NANOS_INSTRUMENT (InstrumentBurst i("accelerator#", fpga->getActiveAcc()+1) );
   //set flag to allow new opdate
   fpga->setUpdate(true);
   FPGADD &dd = ( FPGADD & )wd.getActiveDevice();
   ( dd.getWorkFct() )( wd.getData() );
}

void FPGAThread::yield() {
   verbose("FPGA yield");

   //Synchronizing transfers here seems to yield slightly better performance
   ((FPGAProcessor*)runningOn())->getOutTransferList()->syncAll();
   ((FPGAProcessor*)runningOn())->getInTransferList()->syncAll();

   SMPThread::yield();
}

//Sync transfers on idle
void FPGAThread::idle( bool debug ) {
    //TODO:get the number of transfers from config
    //TODO: figure put a sensible default
    int n = FPGAConfig::getIdleSyncBurst();
    ((FPGAProcessor*)runningOn())->getOutTransferList()->syncNTransfers(n);
    ((FPGAProcessor*)runningOn())->getInTransferList()->syncNTransfers(n);
}

int FPGAThread::getPendingWDs() const {
   return _pendingWD.size();
}

void FPGAThread::finishPendingWD( int numWD ) {
   int n = std::min(numWD, (int)_pendingWD.size() );
   for (int i=0; i<n; i++) {
      WD * wd = _pendingWD.front();
      //Scheduler::postOutlineWork( wd, false, this );
      FPGAWorker::postOutlineWork(wd);
      _pendingWD.pop();
   }
}

void FPGAThread::addPendingWD( WD *wd ) {
   _pendingWD.push( wd );
}

void FPGAThread::finishAllWD() {
   while( !_pendingWD.empty() ) {
      WD * wd = _pendingWD.front();
      //Scheduler::postOutlineWork( wd, false, this );
      FPGAWorker::postOutlineWork(wd);
      _pendingWD.pop();
   }
}

