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

#include "mpiprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "mpidd.hpp"

using namespace nanos;
using namespace nanos::ext;

MPIDevice nanos::ext::MPI("MPI");

MPIDD * MPIDD::copyTo(void *toAddr) {
    MPIDD *dd = new (toAddr) MPIDD(*this);
    return dd;
}

bool MPIDD::isCompatibleWithPE(const ProcessingElement *pe ) {    
    int res=MPI_UNEQUAL;
    nanos::ext::MPIProcessor * myPE = (nanos::ext::MPIProcessor *) pe;
    if (_assignedComm!=NULL) MPI_Comm_compare(myPE->getCommunicator(),_assignedComm,&res);
    //If no assigned comm nor rank, it can run on any PE, if only has a unkown rank, match with comm
    //if has both rank and comm, only execute on his PE
    bool resul =(_assignedComm == NULL && _assignedRank == UNKOWN_RANKSRCDST) 
            || (_assignedRank == UNKOWN_RANKSRCDST && res == MPI_IDENT)
            || (myPE->getRank() == _assignedRank && res == MPI_IDENT);
    return resul;
}

