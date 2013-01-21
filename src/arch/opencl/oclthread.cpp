
#include "oclprocessor.hpp"
#include "oclthread.hpp"

using namespace nanos;
using namespace nanos::ext;

//
// OCLLocalThread implementation.
//

void OCLThread::initializeDependent() {
    printf("iniini dependant\n");
    // Since we create an OCLLocalThread for each OCLProcessor, and an
    // OCLProcessor for each OpenCL device, force device initialization here, in
    // order to be executed in parallel.
    OCLProcessor *myProc = static_cast<OCLProcessor *> (myThread->runningOn());
    myProc->initialize();
    printf("ini dependant\n");
}

void OCLThread::runDependent() {    
    WD &wd = getThreadWD();
    setCurrentWD( wd );
    OpenCLDD &dd = static_cast<OpenCLDD &> (wd.activateDevice(OCLDev));

    dd.getWorkFct()(wd.getData());
}

bool OCLThread::inlineWorkDependent(WD &wd) {
    printf("INI PROCESO\n");
   // Now the WD will be inminently run
   wd.start(WD::IsNotAUserLevelThread);

   OpenCLDD &dd = ( OpenCLDD & )wd.getActiveDevice();

   NANOS_INSTRUMENT ( static nanos_event_key_t key = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("user-code") );
   NANOS_INSTRUMENT ( nanos_event_value_t val = wd.getId() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateAndBurst ( NANOS_RUNNING, key, val ) );
   ( dd.getWorkFct() )( wd.getData() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateAndBurst ( key ) );
   return true;
}

void OCLThread::yield() {
    OCLProcessor &proc = *static_cast<OCLProcessor *> (myThread->runningOn());

    proc.execTransfers();

}

void OCLThread::idle() {
    OCLProcessor &proc = *static_cast<OCLProcessor *> (myThread->runningOn());

    proc.execTransfers();

}

//bool OCLLocalThread::checkForAbort(OCLDD::event_iterator i,
//        OCLDD::event_iterator e) {
//    bool abortNeeded = false;
//
//    for (; i != e; ++i)
//        abortNeeded = abortNeeded || **i < 0;
//
//    return abortNeeded;
//}
