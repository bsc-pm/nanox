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

long *  OS::_argc  = 0;
char ** OS::_argv = 0;
OS::ArgumentList OS::_argList;

const OS::ArgumentList & OS::getProgramArguments ()
{
   long *p;
   int i;

   if ( !_argc ) {
      // variables are before environment
      p=( long * )environ;

      // go backwards until we find argc
      p--;

      for ( i = 0 ; *( --p ) != i; i++ );

      _argc = p;

      _argv = ( char ** ) p+1;

      // build vector
      _argList.reserve( *_argc );

      for ( i = 0; i < *_argc; i++ )
         _argList.push_back( new Argument( _argv[i],i ) );
   }

   return _argList;
}

void OS::consumeArgument ( Argument &arg )
{
   _argv[arg._nparam] = 0;
}

void OS::repackArguments ()
{
   int i,hole = 0;

   // find first hole

   for ( i  = 0; i < *_argc; i++ )
      if ( !_argv[i] ) {
         hole=i++;
         break;
      }

   for ( ; i < *_argc; i++ )
      if ( _argv[i] ) {
         _argv[hole]=_argv[i];
         _argv[i]=0;
         hole++;
      }

   if ( hole != 0 )
      *_argc = hole;
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

