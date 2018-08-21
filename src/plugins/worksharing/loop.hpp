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

#include "nanos-int.h"
#include "atomic_decl.hpp"

namespace nanos {
namespace ext {

typedef struct {
   int64_t                   lowerBound;   // loop lower bound
   int64_t                   upperBound;   // loop upper bound
   int64_t                   loopStep;     // loop step
   int64_t                   chunkSize;    // loop chunk size
   int64_t                   numOfChunks;  // number of chunks for the loop
   Atomic<int64_t>           currentChunk; // current chunk ready to execute
   int64_t                   numParticipants; // number of participants
} WorkSharingLoopInfo;

} // namespace ext
} // namespace nanos
