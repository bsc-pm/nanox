/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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
#include "mpiprocessor_decl.hpp"
#include "mpithread.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "system.hpp"
#include <iostream>
#include <sched.h>
#include <unistd.h>
#include "instrumentation.hpp"
#include <os.hpp>

#include "finish.hpp"

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

    // Set up MPIProcessor and issue taskEnd message
    // reception.
    MPIProcessor& remote = *_runningPEs.at(_currentPE);
    remote.setCurrExecutingWd(&wd);
    remote.getTaskEndRequest().start();

    (dd.getWorkFct())(wd.getData());

    //Check if any task finished
    checkTaskEnd();

    NANOS_INSTRUMENT(sys.getInstrumentation()->raiseCloseStateAndBurst(key, val));
    return false;
}

//Switch to next PE (Round robin way of change PEs to query scheduler)
bool MPIThread::switchToNextFreePE(int uuid){
   int startPE   = _currentPE;
   int currentPE = _currentPE;
   bool switchedCorrectly=false;

   do {
       currentPE = (currentPE+1) % _runningPEs.size();
       switchedCorrectly = switchToPE(currentPE,uuid);
   } while ( !switchedCorrectly && currentPE != startPE );

   return switchedCorrectly;
}

//Switch to PE, under request, if we see a task for this group of PEs
//and the PE is not busy, we can run on that PE and execute the task
bool MPIThread::switchToPE(int rank, int uuid){
    bool ret=false;
    if (rank>=(int)_runningPEs.size()){
        fatal0("You have assigned a rank (" << rank << ") in onto clause to a node which was not allocated before"
                ", your communicator has " << _runningPEs.size() << " processes, possible ranks are [," << (_runningPEs.size()-1) << "], check your code"
                " and make sure that your request finished correctly");
    }
    //In multithread this "test&SetBusy" must be safe
    if (_runningPEs.at(rank)->acquire(uuid) ) {
        _currentPE = rank;
        setRunningOn(_runningPEs.at(_currentPE));
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
    _currentPE = 0;
}

void MPIThread::idle( bool debug ) {
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

Atomic<unsigned int>* MPIThread::getSelfCounter() {
    return &_selfTotRunningWds;
}

void MPIThread::setGroupCounter(Atomic<unsigned int>* gCounter) {
    _groupTotRunningWds=gCounter;
}


std::vector<MPIThread*>* MPIThread::getSelfThreadList(){
    return &_threadList;
}

void MPIThread::setGroupThreadList(std::vector<MPIThread*>* threadList){
    _groupThreadList=threadList;
}

std::vector<MPIThread*>* MPIThread::getGroupThreadList(){
    return _groupThreadList;
}

bool MPIThread::deleteWd(WD* wd, bool markToDelete) {
    bool removable=false;
    if (_groupLock==NULL || _groupLock->tryAcquire()) {
        removable=true;
        //Check if any thread is executing the WD
        for (std::vector<MPIThread*>::iterator it = _groupThreadList->begin() ; it!=_groupThreadList->end() && removable ; ++it) {
            removable=removable && (wd!=(*it)->getCurrentWD());
        }
        if (_groupLock!=NULL) _groupLock->release();
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

inline void MPIThread::freeWD( WD* finished ) {

    WD* previousWD = getCurrentWD();
    setCurrentWD( *finished );

    //Finish the wd, finish work and destroy wd
    Scheduler::finishWork( finished, true );

    setCurrentWD(*previousWD);

    deleteWd( finished, true );

    (*_groupTotRunningWds)--;
}


void MPIThread::checkCommunicationsCompletion( const std::vector<int>& finishedIds ) {
    //Check which tasks have already finished the communication and release dependencies
    std::vector<int>::const_iterator it;
    for( it = finishedIds.begin(); it != finishedIds.end(); ++it ) {
        if (_runningPEs[*it]->testAllRequests() ) {
           WD* previousWD = getCurrentWD();
           WD* wd = _runningPEs[*it]->getCurrExecutingWd();

           setCurrentWD(*wd);
           wd->releaseInputDependencies();
           setCurrentWD(*previousWD);
        }
    }
}

void MPIThread::checkTaskEnd() {
    std::list<WD*>::iterator it;
    for( it = _wdMarkedToDelete.begin(); it != _wdMarkedToDelete.end(); ++it ) {
        if ( deleteWd(*it,/*WD is already in the list */ false) ) {
            it = _wdMarkedToDelete.erase(it);
        } else {
            ++it;
        }
    }

    // Make a local copy of the task end requests
    // in contiguous storage (necessary for wait_some)
    std::vector<mpi::request> pendingTaskEnd;
    pendingTaskEnd = getPendingTaskEndRequests();

    //Receive every task end message and release dependencies for those tasks (only if there are tasks being executed)
    std::vector<int> finishedPEs = waitFinishedTaskEnd( pendingTaskEnd );
    checkCommunicationsCompletion( finishedPEs );
}

std::vector<mpi::request> MPIThread::getPendingTaskEndRequests() {
    std::vector<mpi::request> pendingTaskEnd;
    pendingTaskEnd.reserve( _runningPEs.size() );

    std::vector<MPIProcessor*>::iterator peIterator;
    for( peIterator = _runningPEs.begin(); peIterator != _runningPEs.end(); ++peIterator ) {
        pendingTaskEnd.push_back( (*peIterator)->getTaskEndRequest() );
    }

    return pendingTaskEnd;
}

std::vector<int> MPIThread::waitFinishedTaskEnd( std::vector<mpi::request>& pendingTaskEnd ) {

    std::vector<int> finishedTaskEndIds;
    finishedTaskEndIds = mpi::request::wait_some( pendingTaskEnd );

    std::vector<int>::iterator taskEndIdIter;
    for( taskEndIdIter = finishedTaskEndIds.begin(); taskEndIdIter != finishedTaskEndIds.end(); ++taskEndIdIter ) {
        _currentPE = *taskEndIdIter;
        MPIProcessor* finishedPE = _runningPEs.at( _currentPE );

        setRunningOn(finishedPE);
        //If received something and not mine, stop until whoever is the owner gets it
        WD* finishedWD = finishedPE->freeCurrExecutingWd();
        message0("Received task end message. Finishing workdescriptor " << finishedWD->getId() );

        freeWD( finishedWD );
    }
    return finishedTaskEndIds;
}

void MPIThread::finish() {
    SMPThread::finish();
    //If I'm the master thread of the group (group counter == self-counter)
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized){
        //while (&_selfTotRunningWds!=0) {
        while (_selfTotRunningWds!=0) {
          checkTaskEnd();
        }
        if ( _groupTotRunningWds == &_selfTotRunningWds ) {
            std::vector<MPIProcessor*>& myPEs = getRunningPEs();
            for (std::vector<MPIProcessor*>::iterator it = myPEs.begin(); it!=myPEs.end() ; ++it) {
                //Only release if we are the owner of the process (once released, we are not the owner anymore)
                MPIProcessor* remote = *it;
                if ( remote->getOwner() )
                {
                    mpi::command::Finish::Requestor command( *remote );
                    command.dispatch();

                    remote->waitAllRequests();
                    remote->setOwner(false);
                }
            }
       }
    }
}
