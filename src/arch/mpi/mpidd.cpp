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

#include "mpiprocessor_decl.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "mpidd.hpp"

using namespace nanos;
using namespace nanos::ext;

MPIDevice nanos::ext::MPI("MPI");
Atomic<int> MPIDD::uidGen=1;
bool MPIDD::_spawnDone=false;
//MPI_Comm OFFL_COMM_ANY=MPI_COMM_SELF;

MPIDD * MPIDD::copyTo(void *toAddr) {
    MPIDD *dd = new (toAddr) MPIDD(*this);
    return dd;
}


void MPIDD::setSpawnDone(bool spawnDone) {
   _spawnDone = spawnDone;
}

bool MPIDD::getSpawnDone() {
   return _spawnDone;
}

bool MPIDD::isCompatibleWithPE(const ProcessingElement *pe ) {    
    //PE is null when device gets activated
    if (pe==NULL) return true;
    int res=MPI_UNEQUAL;
    //Only MPI threads will enter this function
    nanos::ext::MPIThread * mpiThread = (nanos::ext::MPIThread *) myThread;
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    if ((uintptr_t)_assignedComm!=0) MPI_Comm_compare(myPE->getCommunicator(),_assignedComm,&res);
        
    //If no assigned comm nor rank, it can run on any PE, if only has a unkown rank, match with comm
    //if has both rank and comm, only execute on his PE    
    bool resul = (((uintptr_t)_assignedComm==0 && _assignedRank<(int)mpiThread->getRunningPEs().size())) 
            || (_assignedRank == UNKOWN_RANKSRCDST && res == MPI_IDENT)
            || (myPE->getRank() == _assignedRank && res == MPI_IDENT);
    
    //If compatible, set the device as busy (if possible) and reserve it for this DD
    resul = resul && myPE->testAndSetBusy(uid, mpiThread->getGroupThreadList()->size() > 1);
    
    //If our current PE is not the right one for the task, check if the right one is free
    if ( !resul && myPE->getRank()!=_assignedRank && ( res == MPI_IDENT || 
            ((uintptr_t)_assignedComm==0 && _assignedRank<(int)mpiThread->getRunningPEs().size())) ){  
       if (_assignedRank==UNKOWN_RANKSRCDST) {
         resul=mpiThread->switchToNextFreePE(uid);      
       } else {
         resul=mpiThread->switchToPE(_assignedRank,uid); 
       }
    } 
    //After we reserve a PE, bind this DD to that PE
    if (resul) {
        _assignedRank= ((MPIProcessor*) mpiThread->runningOn())->getRank();
        _assignedComm= ((MPIProcessor*) mpiThread->runningOn())->getCommunicator();
    }
    return resul;
}

