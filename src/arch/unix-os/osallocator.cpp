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

#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
#include <list>
#include <iostream>
#include <errno.h>

#include "osallocator_decl.hpp"

#define MINIMUM_START_ADDRESS (0x10000)

using namespace nanos;

size_t OSAllocator::computeFreeSpace( uintptr_t start, uintptr_t end, char &unit ) const {
   size_t size = end - start;
   size_t scale = 1;
   char scaleChars[7] = { ' ', 'k', 'M', 'G', 'T', 'P', 'E' };
   unsigned int charIndex = 0;

   while ( ( scale * 1024 ) > 0 && ( size / ( scale * 1024 ) ) > 0 ) {
      scale *= 1024;
      charIndex++;
   }

   unit = scaleChars[ charIndex ];
   return size / scale;
}

uintptr_t OSAllocator::lookForAlignedAddress( size_t len ) const {
   bool found = false;
   uintptr_t target = 0;
   size_t alignedLen;
   unsigned int count = 0;
   while ( (len >> count) != 1 ) count++;

   alignedLen = (1UL<<(count));
 
   std::list< OSMemoryMap >::const_iterator it;
   for (it = freeMaps.begin(); it != freeMaps.end() && !found; it++ )
   {
      if ( it->start < MINIMUM_START_ADDRESS ) continue;
      if ( len < ( it->end - it->start ) ) { //try to find a propper alignment
         uintptr_t possibleTarget = ( it->start & ~( alignedLen-1 ) ) + alignedLen;
         if ( ( possibleTarget + len ) <= it->end ) {
            target = possibleTarget;
            found = true;
         }
      }
   }
   //fprintf(stderr, "selected addr is %p, len %lX, aligned to %lX\n", (void *) target, len, alignedLen);
   return target;
}

int OSAllocator::tryAlloc( uintptr_t addr, size_t len, int flags ) const {
   void *result = mmap( (void *) addr, len, flags, MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED, -1, 0 );
   if ( result == MAP_FAILED ) {
      fprintf(stderr, "mmap failed: %s\n", strerror(errno) );
      return -1;
   } else {
      //fprintf(stderr, "mmap succeded at addr %p, 0x%lx bytes\n", result, len );
      return 0;
   }
}

void OSAllocator::print_current_maps(void) const
{
   char filename[256];
   int ret = 1;
   int fd;
   pid_t mypid = getpid();
   size_t wres;

   sprintf(filename, "/proc/%d/maps", mypid);
   fd = open(filename, O_RDONLY);
   do{
      ret = read(fd, filename, 256);
      wres=write(1, filename, ret);
   } while (ret != 0 && wres==0); 
   close(fd);
}

void OSAllocator::print_parsed_maps() const {
   std::list< OSMemoryMap >::const_iterator it;
   for (it = processMaps.begin(); it != processMaps.end(); it++ )
   {
      fprintf(stderr, "%16lx-%16lx\n", (unsigned long int)(it->start), (unsigned long int)(it->end) );
   }
}

void OSAllocator::print_parsed_maps_full() const {
   uintptr_t current = 0;
   uintptr_t last = -1UL;
   std::list< OSMemoryMap >::const_iterator it;
   char unit;
   for (it = processMaps.begin(); it != processMaps.end(); it++ )
   {
      if ( current < it->start ) {
         size_t len = computeFreeSpace( current, it->start, unit);
         fprintf(stderr, "%16lx-%16lx [free %4lx %cb]\n", (unsigned long int)(current), (unsigned long int)(it->start), (unsigned long int)len, unit );
      }
      fprintf(stderr, "%16lx-%16lx\n", (unsigned long int)it->start, (unsigned long int)it->end );
      current = it->end;
   }
   if ( last > current ) {
      size_t len = computeFreeSpace( current, last, unit);
      fprintf(stderr, "%16lx-%16lx [free %4lx %cb]\n", (unsigned long int)current, (unsigned long int)last, (unsigned long int)len, unit );
   }
}

void OSAllocator::readeMaps()
{
   char unit, filename[256];
   int ret = 1;
   FILE *f;
   unsigned long start, end, num1, num2;
   char str1[64], str2[64], str3[1024], str4[1024];
   pid_t mypid = getpid();
   OSMemoryMap currentMap;
   uintptr_t current = 0;
   uintptr_t last = -1UL;

   str1[0] = '\0';
   str2[0] = '\0';
   str3[0] = '\0';
   str4[0] = '\0';

   sprintf(filename, "/proc/%d/maps", mypid);
   f = fopen(filename, "r");
   if (f == NULL)
   {
      fprintf(stderr, "error opening %s\n", filename);
      return;
   }

   //fprintf(stderr, "maps file %s:\n", filename);
   while (!feof(f))
   {
      ret = fscanf(f, "%lx-%lx %s %lx %s %ld%[ ]%[^\n]", &start, &end, str1, &num1, str2, &num2, str3, str4);

      //fprintf(stderr, "ret %d: %p-%p %s %p %s %ld %s\n", ret, start, end, str1, num1, str2, num2, str4);

      if (ret == 7 || ret == 8)
      {
         currentMap.start = start;
         currentMap.end = end;
         currentMap.prots = 0;
         if (strncmp("rwx", str1, 3) == 0)
         {
            currentMap.prots = PROT_READ | PROT_WRITE | PROT_EXEC;
         }
         else if (strncmp("r-x", str1, 3) == 0)
         {
            currentMap.prots = PROT_READ | PROT_EXEC;
         }
         else if (strncmp("rw-", str1, 3) == 0)
         {
            currentMap.prots = PROT_READ | PROT_WRITE | PROT_EXEC;
         }
         else if (strncmp("r--", str1, 3) == 0)
         {
            currentMap.prots = PROT_READ | PROT_EXEC;
         }
         else if (strncmp("-w-", str1, 3) == 0)
         {
            currentMap.prots = PROT_READ | PROT_WRITE | PROT_EXEC;
         }
         else if (strncmp("--x", str1, 3) == 0)
         {
            currentMap.prots = PROT_READ | PROT_EXEC;
         }
         else if (strncmp("-wx", str1, 3) == 0)
         {
            currentMap.prots = PROT_READ | PROT_WRITE | PROT_EXEC;
         }

         size_t len = computeFreeSpace( current, currentMap.start, unit );
         if ( len > 0 ) {
            //fprintf(stderr, "%16lx-%16lx [free %4ld %cb]\n", current, currentMap.start, len, unit );
            freeMaps.push_back( OSMemoryMap( current, currentMap.start, 0 ) );
         }
         current = currentMap.end;
         processMaps.push_back( currentMap );

#if 0
         if (ret == 8)
         {
            fprintf(stderr, "%16lx-%16lx %s (%d) %lx %s %lu %s\n", start, end, str1,
                  currentMap.prots, num1, str2, num2, str4);
         }
         else
         {
            fprintf(stderr, "%16lx-%16lx %s (%d) %lx %s %lu\n", start, end, str1,
                  currentMap.prots, num1, str2, num2);
         }
#endif
      }
   }
   if ( last > current ) {
      size_t len = computeFreeSpace( current, last, unit );
      if ( len > 0 ) {
         //fprintf(stderr, "%16lx-%16lx [free %4ld %cb]\n", current, last, len, unit );
         freeMaps.push_back( OSMemoryMap( current, currentMap.start, 0 ) );
      }
   }
   fclose(f);
   //fprintf(stderr, "End of process memory maps:\n");
}

void *OSAllocator::allocate( size_t len ) {
   return _allocate( len, false );
}

void *OSAllocator::allocate_none( size_t len ) {
   return _allocate( len, true );
}

void *OSAllocator::_allocate( size_t len, bool none ) {
   uintptr_t targetAddr = 0;
   void *allocatedAddr = NULL;
   readeMaps();
   size_t realLen = len < 4096 ? 4096 : len;
   targetAddr = lookForAlignedAddress( realLen );
   if ( targetAddr != 0 ) {
      if ( tryAlloc( targetAddr, realLen, none ? PROT_NONE : PROT_READ|PROT_WRITE ) ) {
         std::cerr << "mmap failed to allocate " << realLen << " size-aligned bytes." << std::endl;
      } else {
         allocatedAddr = (void *) targetAddr;
      }
   } else {
      std::cerr << "Unable to find a free chunk to allocate " << realLen << " size-aligned bytes." << std::endl;
   }
   processMaps.clear();
   freeMaps.clear();
   return allocatedAddr;
}
