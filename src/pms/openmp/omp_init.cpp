/*************************************************************************************/
/*      Copyright 2010 Barcelona Supercomputing Center                               */
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

#include "system.hpp"
#include <cstdlib>
#include "config.hpp"
#include "omp_data.hpp"

using namespace nanos;

namespace nanos
{
   namespace OpenMP {
      int * ssCompatibility __attribute__( ( weak ) );
      OmpState *globalState;

      static void readEnvinroment ()
      {
         Config config;

         config.setOptionsSection("OpenMP specific");

         // OMP_SCHEDULE
         // OMP_NUM_THREADS
         // OMP_DYNAMIC
         // OMP_NESTED
         // OMP_STACKSIZE
         // OMP_WAIT_POLICY
         // OMP_MAX_ACTIVE_LEVELS
         // OMP_THREAD_LIMIT
         
         config.init();
      }

      static void ompInit()
      {
         // Must be allocated through new to avoid problems with the order of
         // initialization of global objects
         globalState = new OmpState();
         
         if ( ssCompatibility != NULL ) {
            sys.setInitialMode( System::POOL );
         } else {
            sys.setInitialMode( System::ONE_THREAD );
         }

         readEnvinroment();
      }
   }

   System::Init externInit = OpenMP::ompInit;
}

