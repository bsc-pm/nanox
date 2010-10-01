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

#include "gpudd.hpp"
#include "system.hpp"

#include <cuda_runtime.h>

using namespace nanos;
using namespace nanos::ext;

GPUDevice nanos::ext::GPU( "GPU" );

int GPUDD::_gpuCount = 0;
bool GPUDD::_prefetch = true;
bool GPUDD::_overlap = true;
bool GPUDD::_overlapInputs = true;
bool GPUDD::_overlapOutputs = true;
size_t GPUDD::_maxGPUMemory = 0;


GPUDD * GPUDD::copyTo ( void *toAddr )
{
   GPUDD *dd = new ( toAddr ) GPUDD( *this );
   return dd;
}

void GPUDD::printConfiguration()
{
   verbose0( "--- GPUDD configuration ---" );
   verbose0( "  Number of GPU's: " << _gpuCount );
   verbose0( "  Prefetching: " << (_prefetch ? "Enabled" : "Disabled") );
   verbose0( "  Overlapping: " << (_overlap ? "Enabled" : "Disabled") );
   verbose0( "  Overlapping inputs: " << (_overlapInputs ? "Enabled" : "Disabled") );
   verbose0( "  Overlapping outputs: " << (_overlapOutputs ? "Enabled" : "Disabled") );
   if ( _maxGPUMemory != 0 ) {
      verbose0( "  Limited memory: Enabled: " << _maxGPUMemory << " bytes" );
   }
   else {
      verbose0( "  Limited memory: Disabled" );
   }

   verbose0( "--- end of GPUDD configuration ---" );
}

