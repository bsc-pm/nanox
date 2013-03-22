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
    ((MPIProcessor *) myThread->runningOn())->configureCache(MPIProcessor::getCacheDefaultSize(), MPIProcessor::getCachePolicy());
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
    (dd.getWorkFct())(wd.getData());
    NANOS_INSTRUMENT(sys.getInstrumentation()->raiseCloseStateAndBurst(key));
    return true;
}
