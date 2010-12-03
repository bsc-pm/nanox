#include "nanos_omp.h"
#include "omp_data.hpp"
#include "basethread.hpp"

using namespace nanos;
using namespace nanos::OpenMP;

nanos_err_t nanos_omp_single ( bool *b )
{
    OmpData *data = (OmpData *) myThread->getCurrentWD()->getInternalData();
    
    if ( data->isImplicit() ) return nanos_single_guard(b);

    *b=true;
    return NANOS_OK;
}
