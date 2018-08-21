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

/*! \file nanos_sys.cpp
 *  \brief 
 */
#include "nanos.h"
#include "config.h"
#include "cpuset.hpp"
#include "system.hpp"
#include "instrumentationmodule_decl.hpp"
#include "debug.hpp"

// atexit
#include <stdlib.h>

using namespace nanos;

NANOS_API_DEF(const char *, nanos_get_runtime_version, () )
{
   return PACKAGE_VERSION;
}
NANOS_API_DEF(const char *, nanos_get_default_architecture, ())
{
   return (sys.getDefaultArch()).c_str();
}

NANOS_API_DEF(const char *, nanos_get_pm, ())
{
   return (sys.getPMInterface()).getDescription().c_str();
}

NANOS_API_DEF(nanos_err_t, nanos_get_default_binding, ( bool *res ))
{
   try {
      *res = sys.getSMPPlugin()->getBinding();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_get_binding, ( nanos_cpu_set_t *mask ))
{
   try {
      sys.getSMPPlugin()->getCpuProcessMask().copyTo( (cpu_set_t *) mask );
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }
   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_delay_start, ())
{
   try {
      sys.setDelayedStart(true);
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_start, ())
{
   try {
      sys.start();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_finish, ()) 
{
   try {
      sys.finish();
   } catch ( ... ) { 
      return NANOS_UNKNOWN_ERR;
   }   

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_current_socket, (int socket ))
{
   try {
      sys.setUserDefinedNUMANode( socket );
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_get_num_sockets, (int *num_sockets ))
{
   try {
      *num_sockets = sys.getNumNumaNodes();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}


NANOS_API_DEF(void, ompss_nanox_main_begin, ( void *addr, const char* file, int line ))
{
    sys.ompss_nanox_main(addr, file, line);
}

NANOS_API_DEF(void, ompss_nanox_main_end, ( ))
{
    sys.ompss_nanox_main_end();
}

// Deprecated API
NANOS_API_DEF(void, ompss_nanox_main, ( ))
{    
    warning("This application is using an old instrumentation API, please update your compiler");

    ompss_nanox_main_begin( (void*)ompss_nanox_main, __FILE__, __LINE__);
}

NANOS_API_DEF(void, nanos_atexit, (void *p))
{
    ::atexit((void (*)())p);
}
NANOS_API_DEF(int, nanos_cmpi_init, (int *argc, char **argv[]))
{
   return sys.initClusterMPI(argc, argv);
}

NANOS_API_DEF(void, nanos_cmpi_finalize, (void))
{
   sys.finalizeClusterMPI();
}

NANOS_API_DEF(void, nanos_into_blocking_mpi_call, (void))
{
   sys.notifyIntoBlockingMPICall();
}

NANOS_API_DEF(void, nanos_out_of_blocking_mpi_call, (void))
{
   sys.notifyOutOfBlockingMPICall();
}

NANOS_API_DEF(void, nanos_thread_print, (char *str))
{
   *myThread->_file << str << std::flush;
}
NANOS_API_DEF(void, nanos_set_watch_addr, (void *addr))
{
   sys._watchAddr = addr;
}
NANOS_API_DEF(void, nanos_print_bt, (void))
{
   printBt(std::cerr);
}

NANOS_API_DEF(void, nanos_enable_verbose_copies, (void))
{
   sys.setVerboseCopies(true);
   sys.setVerboseDevOps(true);
}
NANOS_API_DEF(void, nanos_disable_verbose_copies, (void))
{
   sys.setVerboseCopies(false);
   sys.setVerboseDevOps(false);
}
