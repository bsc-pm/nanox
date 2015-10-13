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
    _ranksWithPendingComms.push_back(_currPe);
    
    (dd.getWorkFct())(wd.getData());    
    
    //Check if any task finished
    checkTaskEnd();
    
    NANOS_INSTRUMENT(sys.getInstrumentation()->raiseCloseStateAndBurst(key, val));
    return false;
}

//Switch to next PE (Round robin way of change PEs to query scheduler)
bool MPIThread::switchToNextFreePE(int uuid){
   int startPE=_currPe;
	int currPe=_currPe;
	bool switchedCorrectly=false;
	
   do {        
       currPe=(currPe+1)%_runningPEs.size();
	    switchedCorrectly=switchToPE(currPe,uuid);
   } while ( !switchedCorrectly && currPe!=startPE );
		
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
    //In multithread this "test&SetBusy" bust be safe
    if (_runningPEs.at(rank)->testAndSetBusy(uuid, _groupThreadList->size()>1)) {
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

inline void MPIThread::freeCurrExecutingWD(MPIProcessor* finishedPE){    
    //Receive the signal (just to remove it from the MPI "queue")
    int id_func_ompss;
    nanos::ext::MPIRemoteNode::nanosMPIRecvTaskend(&id_func_ompss, 1, MPI_INT,finishedPE->getRank(),
        finishedPE->getCommunicator(),MPI_STATUS_IGNORE);
    WD* wd=finishedPE->getCurrExecutingWd();
    finishedPE->setCurrExecutingWd(NULL);
    WD* previousWD = getCurrentWD();
    setCurrentWD(*wd);
    //Clear all async requests on this PE (they finished a while ago)
    finishedPE->clearAllRequests();    
    wd->finish();
    //Set the PE as free so we can re-schedule work to it
    finishedPE->setBusy(false);
    //Finish the wd, finish work and destroy wd
    Scheduler::finishWork(wd,true);
    setCurrentWD(*previousWD);
    deleteWd(wd,true);
    (*_groupTotRunningWds)--;
}


void MPIThread::checkCommunicationsCompletion() {  
    //Check which tasks have already finished the communication and release dependencies
    std::list<int>::iterator iter=_ranksWithPendingComms.begin();
    while( iter!=_ranksWithPendingComms.end() ){
      int rank=*iter;
      if (_runningPEs[rank]->testAllRequests()) { 
         iter=_ranksWithPendingComms.erase(iter);
         WD* previousWD = getCurrentWD();
         WD* wd=_runningPEs[rank]->getCurrExecutingWd();
         setCurrentWD(*wd);
         wd->releaseInputDependencies();
         setCurrentWD(*previousWD);
      } else {
         ++iter;
      }
    }
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
    unsigned int spinCounter=0;
    //Receive every task end message and release dependencies for those tasks (only if there are tasks being executed)
    if (*_groupTotRunningWds!=0 && (_groupLock==NULL || _groupLock->tryAcquire())){ 
        //If every node is busy, no need to search new tasks, we cam stay checking if any task finished
        do {
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_END_TASK,((MPIProcessor *) myThread->runningOn())->getCommunicator(), &flag, 
                       &status);
            if (flag!=0) {
                MPIProcessor* finishedPE=_runningPEs.at(status.MPI_SOURCE);
                _currPe=status.MPI_SOURCE;
                setRunningOn(finishedPE);
                //If received something and not mine, stop until whoever is the owner gets it
                freeCurrExecutingWD(finishedPE);
            }
            spinCounter++;    
        } while ( _groupTotRunningWds->value()==getRunningPEs().size() && spinCounter<2000);
        
        spinCounter=0;
        //If every node is busy, no need to search new tasks, we cam stay checking if any task finished
        while ( _groupTotRunningWds->value()==getRunningPEs().size() ) {
            MPI_Iprobe(MPI_ANY_SOURCE, TAG_END_TASK,((MPIProcessor *) myThread->runningOn())->getCommunicator(), &flag, 
                       &status);
            if (flag!=0) {
                MPIProcessor* finishedPE=_runningPEs.at(status.MPI_SOURCE);
                _currPe=status.MPI_SOURCE;
                setRunningOn(finishedPE);
                //If received something and not mine, stop until whoever is the owner gets it
                freeCurrExecutingWD(finishedPE);
            }
            spinCounter++;
            unsigned int usec=100*(spinCounter/50);
            if (usec>500000) { 
                checkCommunicationsCompletion();
                usec=500000;
            }
            usleep(usec);       
        } 
        if (_groupLock!=NULL) _groupLock->release();
    } else if (*_groupTotRunningWds!=0) {
        //If every node is busy, no need to search new tasks, we cam stay checking if any task finished
        //We only sleep here, as (if exists) other thread will be reciving tasks
        while ( _groupTotRunningWds->value()==getRunningPEs().size() ) {
            spinCounter++;
            unsigned int usec=100*(spinCounter/50);
            if (usec>500000) {
                checkCommunicationsCompletion();
                usec=500000;
            }
            usleep(usec);       
        }        
    }
    checkCommunicationsCompletion();
}

void MPIThread::finish() {
    //If I'm the master thread of the group (group counter == self-counter)
    int resul;
    MPI_Finalized(&resul);
    if (!resul){
        while (&_selfTotRunningWds!=0) {
          checkTaskEnd();
        }
        if ( _groupTotRunningWds == &_selfTotRunningWds ) {
            cacheOrder order;
            order.opId = OPID_FINISH;
            std::vector<MPIProcessor*>& myPEs = getRunningPEs();
            for (std::vector<MPIProcessor*>::iterator it = myPEs.begin(); it!=myPEs.end() ; ++it) {
                //Only release if we are the owner of the process (once released, we are not the owner anymore)
                if ( (*it)->getOwner() ) 
                {
                    nanos::ext::MPIRemoteNode::nanosMPISsend(&order, 1, nanos::MPIDevice::cacheStruct, (*it)->getRank(), TAG_M2S_ORDER, (*it)->getCommunicator());
                    (*it)->setOwner(false);
                }
            }
       }
    }
}
