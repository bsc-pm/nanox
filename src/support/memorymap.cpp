#ifdef STANDALONE_TEST

#ifdef message
#undef message
#define message(x)
#else
#define message(x)
#endif
#ifdef ensure
#undef ensure
#define ensure(x,y)
#else
#define ensure(x,y)
#endif
#ifndef NEW
#define NEW new
#endif

#endif
#include "memorymap.hpp"

const char* nanos::MemoryChunk::strOverlap[] = {
   "NO_OVERLAP",
   "BEGIN_OVERLAP",
   "END_OVERLAP",
   "TOTAL_OVERLAP",
   "SUBCHUNK_OVERLAP",
   "TOTAL_BEGIN_OVERLAP",
   "SUBCHUNK_BEGIN_OVERLAP",
   "TOTAL_END_OVERLAP",
   "SUBCHUNK_END_OVERLAP"
};
