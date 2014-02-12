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
/*! \file nanos_sys.cpp
 *  \brief 
 */
#include "nanos.h"
#include "system.hpp"
#include "instrumentationmodule_decl.hpp"

using namespace nanos;

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
      *res = sys.getBinding();
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
      sys.setCurrentSocket( socket );
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

NANOS_API_DEF(nanos_err_t, nanos_get_num_sockets, (int *num_sockets ))
{
   try {
      *num_sockets = sys.getNumAvailSockets();
   } catch ( ... ) {
      return NANOS_UNKNOWN_ERR;
   }

   return NANOS_OK;
}

//This main will do nothing normally
//It will act as an slave and call exit(0) when we need slave behaviour
//in offload or cluster version
NANOS_API_DEF(void, ompss_nanox_main, ( ))
{    
    sys.ompss_nanox_main();    
}