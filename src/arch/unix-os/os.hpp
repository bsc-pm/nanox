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

namespace nanos
{

// this is UNIX-like OS
// TODO: ABS and virtualize

   class OS
   {

      public:

         class Argument
         {

               friend class OS;

            public:
               char *   _name;
               int      _nparam;

            public:
               Argument( char *arg,int i ) : _name( arg ),_nparam( i ) {}

               char * getName() const { return _name; }
         };

         //TODO: make it autovector?
         typedef std::vector<Argument *> ArgumentList;

         static char **          _argv;
         static long *           _argc;
         static ArgumentList     _argList;

         static const char *getEnvironmentVariable( const std::string &variable );
         static const ArgumentList & getProgramArguments ();
         static void consumeArgument ( Argument &arg );
         static void repackArguments ();

         static void * loadDL( const std::string &dir, const std::string &name );
         static void * dlFindSymbol( void *dlHandler, const std::string &symbolName );
         static void * dlFindSymbol( void *dlHandler, const char *symbolName );
         // too-specific?
         static char * dlError( void *dlHandler ) { return dlerror(); }
   };

// inlined functions

   inline const char * OS::getEnvironmentVariable ( const std::string &name )
   {
      return getenv( name.c_str() );
   }

};


#endif

