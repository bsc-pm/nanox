#ifndef _NANOS_MEMTRACKER
#define _NANOS_MEMTRACKER

#if 0
#ifdef NANOS_DEBUG_ENABLED

#include <new>

void* operator new ( size_t size );
void* operator new ( size_t size, const char *file, int line );
void* operator new[] ( size_t size, const char *file, int line );
void operator delete ( void *p );
void operator delete ( void *p, char *file, char *line );

#define NEW new(__FILE__, __LINE__)

#else 

#define NEW new

#endif

#endif

#endif

