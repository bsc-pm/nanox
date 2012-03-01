
#include "nanos-int.h"
#include "atomic_decl.hpp"

namespace nanos {
namespace ext {

typedef struct {
   int                   lowerBound;   // loop lower bound
   int                   upperBound;   // loop upper bound
   int                   loopStep;     // loop step
   int                   chunkSize;    // loop chunk size
   int                   numOfChunks;  // number of chunks for the loop
   Atomic<int>           currentChunk; // current chunk ready to execute
} WorkSharingLoopInfo;

#if 0
   int                   neths;    // additional data to expand team
   nanos_thread_t       *eths;     // additional data to expand team
#endif

} // namespace ext
} // namespace nanos
