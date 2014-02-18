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

#include "mpi.h"
#include "mpiprocessor.hpp"
#include "mpithread.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "system.hpp"
#include <iostream>
#include <sched.h>
#include <unistd.h>
#include "instrumentation.hpp"

using namespace nanos;
using namespace nanos::ext;

void MPIThread::initializeDependent() {
}

void MPIThread::runDependent() {
    WD &work = getThreadWD();
    setCurrentWD(work);
    
    MPIDD &dd = (MPIDD &) work.activateDevice(MPI);

    dd.getWorkFct()(work.getData());
}

bool MPIThread::inlineWorkDependent(WD &wd) {
    // Now the WD will be inminently run
    wd.start(WD::IsNotAUserLevelThread);

    MPIDD &dd = (MPIDD &) wd.getActiveDevice();
    NANOS_INSTRUMENT(static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("user-code"));
    NANOS_INSTRUMENT(nanos_event_value_t val = wd.getId());
    NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenStateAndBurst(NANOS_RUNNING, key, val));
    
    (*_groupTotRunningWds)++;
    _runningPEs.at(_currPe)->setCurrExecutingWd(&wd);
    (dd.getWorkFct())(wd.getData());
    //Check if any task finished
    checkTaskEnd();
    
    NANOS_INSTRUMENT(sys.getInstrumentation()->raiseCloseStateAndBurst(key, val));
    return false;
}

//Switch to next PE (Round robin way of change PEs to query scheduler)
void MPIThread::switchToNextPE(){
    int start=_currPe;
    _currPe=(_currPe+1)%_runningPEs.size();
    //This is busy does not need to be thread-safe, it's here just
    //as an optimization
    while (_runningPEs.at(_currPe)->isBusy() && _currPe!=start ){        
        _currPe=(_currPe+1)%_runningPEs.size();
    }
    if (_currPe!=start) setRunningOn(_runningPEs.at(_currPe));
}

//Switch to PE, under request, if we see a task for this group of PEs
//and the PE is not busy, we can run on that PE and execute the task
bool MPIThread::switchToPE(int rank, int uuid){
    bool ret=false;
    if (rank>_runningPEs.size()){
        fatal0("You have assigned a rank in onto clause which was not spawned before"
                ", check your program");
    }
    //In multithread this "test&SetBusy" bust be safe
    if (_runningPEs.at(rank)->testAndSetBusy(uuid)) {
        _currPe=rank;
        setRunningOn(_runningPEs.at(_currPe));
        ret=true;
    }
    return ret;
}


void MPIThread::addRunningPEs( MPIProcessor** pe, int nPes){
    _runningPEs.resize(nPes);
    for (int i=0;i<nPes;i++){
      _runningPEs[pe[i]->getRank()]=pe[i];
    }
    setRunningOn(_runningPEs.at(0));
    _currPe=0;
}

void MPIThread::idle() {
    checkTaskEnd();
}

std::vector<MPIProcessor*>& MPIThread::getRunningPEs() {
    return _runningPEs;
}

void MPIThread::setGroupLock(Lock* gLock) {
    _groupLock=gLock;
}


Lock* MPIThread::getSelfLock() {
    return &_selfLock;
}

void MPIThread::increaseCopyInCount(){
    _numPendingComms++;
}

void MPIThread::resetCopyInCount(){
    _numPendingComms=0;
}

int MPIThread::getCopyInCount(){
    return _numPendingComms;
}
         
Atomic<int>* MPIThread::getSelfCounter() {
    return &_totRunningWds;
}
         
void MPIThread::setGroupCounter(Atomic<int>* gCounter) {
    _groupTotRunningWds=gCounter;
}


std::vector<MPIThread*>* MPIThread::getSelfThreadList(){
    return &_threadList;
}

void MPIThread::setGroupThreadList(std::vector<MPIThread*>* threadList){
    _groupThreadList=threadList;
}

bool MPIThread::deleteWd(WD* wd, bool markToDelete) {
    bool removable=true;
    //Check if any thread is executing the WD
    for (std::vector<MPIThread*>::iterator it = _groupThreadList->begin() ; it!=_groupThreadList->end() && removable ; ++it) {
        removable=removable && (wd!=(*it)->getCurrentWD());
    }
    //Delete the WD (only if we are not executing it, this should be mostly always)
    if (removable) {
        wd->~WorkDescriptor();
        delete[] (char *)wd;
    } else if (markToDelete) {
        _wdMarkedToDelete.push_back(wd);
    }
    return removable;
}

inline void MPIThread::freeCurrExecutingWD(MPIProcessor* finishedPE){    
    //Receive the signal (just to remove it from the MPI "queue")
    int id_func_ompss;
    nanos::ext::MPIProcessor::nanosMPIRecvTaskend(&id_func_ompss, 1, MPI_INT,finishedPE->getRank(),
        finishedPE->getCommunicator(),MPI_STATUS_IGNORE);
    WorkDescriptor* wd=finishedPE->getCurrExecutingWd();
    //Before finishing wd, switch thread to the right PE
    //Wait until local cache in the PE is free
    PE* oldPE=runningOn();
    setRunningOn(finishedPE);
    //Finish the wd, finish work and destroy wd
    wd->finish();
    Scheduler::finishWork(wd,true);
    setRunningOn(oldPE);
    finishedPE->setCurrExecutingWd(NULL);
    finishedPE->setBusy(false);
    deleteWd(wd,true);
    //Restore previous PE
    (*_groupTotRunningWds)--;
}

void MPIThread::checkTaskEnd() {
    for (std::list<WD*>::iterator it=_wdMarkedToDelete.begin(), next ; it!=_wdMarkedToDelete.end(); it=next) {    
        next = it;
        ++next;
        if ( deleteWd(*it,/*WD is already in the list */ false) ) {                
            _wdMarkedToDelete.erase(it);
        }
    }    
    int flag=1;
    MPI_Status status;
    //Receive every task end message and release dependencies for those tasks (only if there are tasks being executed)
    if (*_groupTotRunningWds!=0 && (_groupLock==NULL || _groupLock->tryAcquire())){
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_END_TASK,((MPIProcessor *) myThread->runningOn())->getCommunicator(), &flag, 
                   &status);
        if (flag!=0) {
            MPIProcessor* finishedPE=_runningPEs.at(status.MPI_SOURCE);
            //If received something and not mine, stop until whoever is the owner gets it
            freeCurrExecutingWD(finishedPE);
        }
        if (_groupLock!=NULL) _groupLock->release();
    }
}