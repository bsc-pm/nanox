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

#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "system.hpp"
#include <iostream>

using namespace nanos;

Device nanos::SMP( "SMP" );

int SMPDD::stackSize = 1024;

/*! \fn prepareConfig(Config &config)
  \brief Registers the Device's configuration options
  \param reference to a configuration object.
  \sa Config System
*/
void SMPDD::prepareConfig( Config &config )
{
   /*!
      Get the stack size from system configuration
    */
   stackSize = sys.getDeviceStackSize();

   /*!
      Get the stack size for this device
   */
   config.registerArgOption( new Config::PositiveVar( "nth-smp-stack-size",stackSize ) );
   config.registerEnvOption( new Config::PositiveVar( "NTH_SMP_STACK_SIZE",stackSize ) );
}

void SMPDD::allocateStack ()
{
   stack = new intptr_t[stackSize];
}

void SMPDD::initStack ( void *data )
{
   if ( !hasStack() ) {
      allocateStack();
   }

   initStackDep( ( void * )getWorkFct(),data,( void * )Scheduler::exit );
}
