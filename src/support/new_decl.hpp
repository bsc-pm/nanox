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

#ifndef _NANOS_NEW_DECL
#define _NANOS_NEW_DECL

#include <new>

#if defined(NANOS_DEBUG_ENABLED) && defined(NANOS_MEMTRACKER_ENABLED) // ----- debug AND memtracker -----

#include <cstdlib>

   #define NEW new(__FILE__, __LINE__)

   void* operator new ( size_t size, const char *file, int line );
   void* operator new[] ( size_t size, const char *file, int line );
   
   void operator delete ( void *p, const char *file, int line );
   void operator delete[] ( void *p, const char *file, int line );
 
#else // ----- no memtracker ------

#define NEW new

#endif // ----- all versions -----

#endif // _NANOS_NEW_DECL
