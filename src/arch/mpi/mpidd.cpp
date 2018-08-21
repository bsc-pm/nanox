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

#include "mpiprocessor_decl.hpp"
#include "mpithread.hpp"
#include "mpidd.hpp"

#include "schedule.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "instrumentation.hpp"

using namespace nanos;
using namespace nanos::ext;

MPIDevice nanos::ext::MPI("MPI");
Atomic<int> MPIDD::uidGen=1;
bool MPIDD::_spawnDone=false;

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
    if( pe == NULL )
        return true;

    //Only MPI threads will enter this function
    MPIThread* thread = (MPIThread*) myThread;
    MPIProcessor* remote = (MPIProcessor*) pe;

    bool isCompatible = true;
    if( static_cast<int>(_assignedComm) != 0 ) {
        int res = MPI_UNEQUAL;
        MPI_Comm_compare( remote->getCommunicator(),_assignedComm, &res );
        isCompatible &= res == MPI_IDENT;
    }
    isCompatible &= _assignedRank == UNKNOWN_RANK || _assignedRank == remote->getRank();

    //If compatible, set the device as busy (if possible) and reserve it for this DD
    bool reserved = false;
    if( isCompatible )
        reserved = remote->acquire(uid);

    //If our current PE is not the right one for the task, check if the right one is free
    if( !reserved ) {
       if( _assignedRank == UNKNOWN_RANK ) {
         reserved = thread->switchToNextFreePE( uid );
       } else {
         reserved = thread->switchToPE( _assignedRank, uid );
       }
    }
    //After we reserve a PE, bind this DD to that PE
    if( reserved ) {
	remote = static_cast<MPIProcessor*>(thread->runningOn());
        _assignedRank = remote->getRank();
        _assignedComm = remote->getCommunicator();
    }
    return reserved;
}

