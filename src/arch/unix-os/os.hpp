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

#ifndef _NANOS_OS
#define _NANOS_OS

#include <string>
#include <vector>
#include <stdlib.h>
#include <dlfcn.h>
#include <time.h>

namespace nanos
{

// this is UNIX-like OS
// TODO: ABS and virtualize

   class OS
   {
      // All members are static so we don't need a constructor/destructor/...
      
         static long _argc; 
         static char ** _argv; 
      public:

         static void init ();

         static const char *getEnvironmentVariable( const std::string &variable );

         static void * loadDL( const std::string &dir, const std::string &name );
         static void * dlFindSymbol( void *dlHandler, const std::string &symbolName );
         static void * dlFindSymbol( void *dlHandler, const char *symbolName );
         // too-specific?
         static char * dlError( void *dlHandler ) { return dlerror(); }

         static const char * getArg (int i) { return _argv[i]; }
         static long getArgc() { return _argc; }

         static double getMonotonicTime ();
	 static double getMonotonicTimeResolution ();
   };

// inlined functions

   inline const char * OS::getEnvironmentVariable ( const std::string &name )
   {
      return getenv( name.c_str() );
   }

   inline double OS::getMonotonicTime ()
   {
      struct timespec ts;
      double t;

      clock_gettime(CLOCK_MONOTONIC,&ts);

      t = (double) (ts.tv_sec)  + (double) ts.tv_nsec * 1.0e-9;

      return t;
   }

   inline double OS::getMonotonicTimeResolution ()
   {
      struct timespec ts;
      double res;

      clock_getres(CLOCK_MONOTONIC,&ts);

      res = (double) (ts.tv_sec)  + (double) ts.tv_nsec * 1.0e-9;

      return res;
   }

};


#endif

