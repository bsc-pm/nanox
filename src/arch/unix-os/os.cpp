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

#include "os.hpp"
#include <stdlib.h>

using namespace nanos;

long OS::_argc = 0; 
char ** OS::_argv = 0; 

static void findArgs (long *argc, char ***argv) 
{
   long *p; 
   int i; 

   // variables are before environment 
   p=( long * )environ; 

   // go backwards until we find argc 
   p--; 

   for ( i = 0 ; *( --p ) != i; i++ ); 

   *argc = *p; 
   *argv = ( char ** ) p+1; 
}

void OS::init ()
{
   findArgs(&_argc,&_argv);
}

void * OS::loadDL( const std::string &dir, const std::string &name )
{
   std::string filename;
   filename = dir + "/" + name + ".so";
   /* open the module */
   return dlopen ( filename.c_str(), RTLD_NOW );
}

void * OS::dlFindSymbol( void *dlHandler, const std::string &symbolName )
{
   return dlsym ( dlHandler, symbolName.c_str() );
}

void * OS::dlFindSymbol( void *dlHandler, const char *symbolName )
{
   return dlsym ( dlHandler, symbolName );
}

