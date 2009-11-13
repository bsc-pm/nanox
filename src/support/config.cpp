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

#include <iostream>
#include <sstream>
#include <string>
#include "config.hpp"
#include "os.hpp"
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <ext/functional>
#include <functional>
#include "debug.hpp"
#include "functors.hpp"

using namespace nanos;

void Config::setDefaults()
{
}

void Config::parseFiles ()
{
}

void Config::registerEnvOption ( Option *opt )
{
   _envOptions.push_back( opt );
}

void Config::registerArgOption ( Option *opt )
{
   _argOptions[opt->getName()] = opt;
}

void Config::parseEnvironment ()
{
   for ( OptionList::iterator it = _envOptions.begin();
         it < _envOptions.end(); it++ ) {
      Option &opt = **it;

      const char *env = OS::getEnvironmentVariable( opt.getName() );

      if ( !env ) continue;

      const std::string tmp( env );

      std::istringstream iss( env );

      try {
         opt.parse( env );
      } catch ( InvalidOptionException &exception ) {
         std::cerr << "WARNING:" << exception.what() << std::endl;
      }
   }
}

// C-strings are used in this function because std::string substring
// and character removal are O(N) compared to the possible O(1)
// with C style pointer manipulation.
// Even so, a std::string needs to be constructed to access the argument map
// so it's not clear if its worth it.
void Config::parseArguments ()
{
   const OS::ArgumentList & list = OS::getProgramArguments();

   for ( OS::ArgumentList::const_iterator it = list.begin();
         it < list.end(); it++ ) {

      char * arg( ( *it )->_name );
      char * value=0;
      bool needValue=true;

      if ( arg[0] != '-' ) continue;

      arg++;

      // support --args
      if ( arg[0] == '-' ) arg++;

      if ( ( value = strchr( arg,'=' ) ) != NULL ) {
         // -arg=value form
         *value = 0; // sepparate arg from value
         value++; // point to the beginning of value
         needValue = false;
      } else {
         // -arg value form
      }

      OptionMap::iterator obj = _argOptions.find( std::string( arg ) );

      if ( obj != _argOptions.end() ) {
         Option &opt = *( *obj ).second;

         if ( needValue && opt.getType() != Option::FLAG ) {
            OS::consumeArgument( *( *it ) );
            it++;

            if ( it == list.end() )
               throw InvalidOptionException( opt,"" );

            value = ( *it )->_name;
         }

         try {
            opt.parse( value );
         } catch ( InvalidOptionException &exception ) {
            std::cerr << "WARNING:" << exception.what() << std::endl;
         }

         OS::consumeArgument( *( *it ) );
      }
   }

   OS::repackArguments();
}

void Config::init ()
{
   setDefaults();
   parseFiles();
   parseEnvironment();
   parseArguments();
}

//TODO: move to utility header

void Config::clear ()
{
   std::for_each( _envOptions.begin(),_envOptions.end(),deleter<Option> );
   std::for_each( _argOptions.begin(),_argOptions.end(),pair_deleter2<Option> );
   _envOptions.clear();
   _argOptions.clear();
}

//TODO: generalize?

class map_copy
{
      Config::OptionMap& dest;

   public:

      map_copy( Config::OptionMap &d ) : dest( d ) {}

      void operator()( Config::OptionMap::value_type pair ) {  dest[pair.first] = pair.second->clone(); }
};

void Config::copy ( const Config &cfg )
{
   std::transform( cfg._envOptions.begin(), cfg._envOptions.end(), _envOptions.begin(),
                   cloner<Option> );
   std::for_each( cfg._argOptions.begin(), cfg._argOptions.end(), map_copy( _argOptions ) );
}

Config::Config ( const Config &cfg )
{
   copy( cfg );
}

const Config & Config::operator= ( const Config &cfg )
{
   // handle self-assignment
   if ( this == &cfg ) return *this;

   clear();

   copy( cfg );

   return *this;
}

Config::~Config ()
{
   clear();
}

/** Options **/

