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
#include <string.h>
#include "config.hpp"
#include "os.hpp"
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <ext/functional>
#include <functional>
#include "debug.hpp"
#include "functors.hpp"
#include <map>

using namespace nanos;


Config::NanosHelp *Config::_nanosHelp = NULL;

void Config::NanosHelp::addHelpString ( const std::string &section, const std::string &argHelpString, const std::string &envHelpString )
{
   _helpSections[section].push_back( std::make_pair( argHelpString, envHelpString ) );
}

void Config::NanosHelp::addSectionDescription ( const std::string &section, const std::string &description )
{
   _sectionDescriptions[section] = description;
}

const std::string Config::NanosHelp::getHelp()
{
   std::stringstream helpString;

   for ( SectionsMap::iterator it = _helpSections.begin(); it != _helpSections.end(); it++ ) {
      helpString << it->first;

      SectionDescriptionsMap::iterator desc = _sectionDescriptions.find ( it->first );
      if ( desc != _sectionDescriptions.end() ) {
         helpString << "\t" << desc->second;
      }

      helpString << std::endl;

      std::stringstream argHelpString;
      std::stringstream envHelpString;
      HelpStringList &optionsHelpList = it->second;
      for ( HelpStringList::iterator it = optionsHelpList.begin(); it != optionsHelpList.end(); it++ ) {
         if ( it->first != "" ) argHelpString << "\t" << it->first << std::endl;
         if ( it->second != "" ) envHelpString << "\t" << it->second << std::endl;
      }

      helpString << "   'NANOS_ARGS' options" << std::endl;
      helpString << argHelpString.str();
      helpString << "   Environment variables" << std::endl;
      helpString << envHelpString.str();
      helpString << std::endl;
   }

   return helpString.str();
}

const std::string Config::getNanosHelp()
{
   return _nanosHelp->getHelp();
}

void Config::setDefaults()
{
}

void Config::parseFiles ()
{
}

void Config::registerEnvOption ( const std::string &option, const std::string &envVar )
{
   _configOptions[option]->setEnvVar( envVar );
}

void Config::registerArgOption ( const std::string &option, const std::string &arg )
{
   _configOptions[option]->setArg( arg );
   _argOptionsMap[arg] = _configOptions[option];
}

void Config::registerConfigOption ( const std::string &optionName, Option *option, const std::string &helpMessage )
{
   ConfigOption *configOption = new ConfigOption( optionName, *option, helpMessage, _currentSection );
   _configOptions[optionName] = configOption;
}

void Config::registerAlias ( const std::string &optionName, const std::string &alias, const std::string &helpMessage )
{
   ConfigOption *option = _configOptions[optionName];
   ConfigOption *aliasOption = new ConfigOption( alias, option->getOption(), helpMessage, _currentSection );
   _configOptions[alias] = aliasOption;
}

void Config::parseEnvironment ()
{
   for ( ConfigOptionMap::iterator it = _configOptions.begin(); it != _configOptions.end(); it++ ) {
      ConfigOption &confOpt = * ( it->second );
      if ( confOpt.getEnvVar() != "" ) {
         Option &opt = confOpt.getOption();
         opt.setName( confOpt.getEnvVar() );

         const char *env = OS::getEnvironmentVariable( confOpt.getEnvVar() );

         if ( !env ) continue;

         try {
            opt.parse( env );
         } catch ( InvalidOptionException &exception ) {
            std::cerr << "WARNING:" << exception.what() << std::endl;
         }
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
   const char *tmp = OS::getEnvironmentVariable( "NANOS_ARGS" );

   if ( tmp == NULL ) return;

   char env[ strlen(tmp) + 1 ];
   strcpy( &env[0], tmp );
   char *arg = strtok( &env[0], " " );

   while ( arg != NULL) {
      char * value=0;
      bool needValue=true;
      bool hasNegation=false;

      if ( arg[0] != '-' ) {
         arg = strtok( NULL, " " );
         continue;
      }

      arg++;

      // support --args
      if ( arg[0] == '-' ) arg++;

      // Since this skips the negation prefix no other arguments
      // are allowed to start with the negation prefix.
      if ( strncmp( arg, "no-", 3 ) == 0) {
         hasNegation=true;
         arg+=3;
      }

      if ( ( value = strchr( arg,'=' ) ) != NULL ) {
         // -arg=value form
         *value = 0; // sepparate arg from value
         value++; // point to the beginning of value
         needValue = false;
      } else {
         // -arg value form
      }

      ConfigOptionMap::iterator obj = _argOptionsMap.find( std::string( arg ) );

      if ( obj != _argOptionsMap.end() ) {
         Option &opt = ( *obj ).second->getOption();

         if ( needValue && opt.getType() != Option::FLAG ) {
            value = strtok( NULL, " " );
            if ( value == NULL)
               throw InvalidOptionException( opt,"" );
         }
         char yes[] = "yes";
         char no[] = "no";

         if ( opt.getType() == Option::FLAG ) {
            if ( hasNegation )
               value = no;
            else
               value = yes;
         }

         try {
            opt.setName( std::string( arg ) );
            opt.parse( value );
         } catch ( InvalidOptionException &exception ) {
            std::cerr << "WARNING:" << exception.what() << std::endl;
         }
      }
      arg = strtok( NULL, " " );
   }
}

void Config::init ()
{
   setDefaults();
   parseFiles();
   parseEnvironment();
   parseArguments();

   if ( _nanosHelp == NULL ) {
      _nanosHelp = new NanosHelp();
   }

   for ( ConfigOptionMap::iterator it = _configOptions.begin(); it != _configOptions.end(); it++ ) {
      _nanosHelp->addHelpString ( (*it).second->getSection(), (*it).second->getArgHelp(), (*it).second->getEnvHelp() );
   }
}

void Config::setOptionsSection( const std::string &sectionName, const std::string *sectionDescription )
{
   _currentSection = sectionName;

   if ( sectionDescription != NULL ) {
      if ( _nanosHelp == NULL ) {
         _nanosHelp = new NanosHelp();
      }
      _nanosHelp->addSectionDescription ( sectionName, *sectionDescription );
   }
}

//TODO: move to utility header

void Config::clear ()
{
   std::for_each( _configOptions.begin(),_configOptions.end(),pair_deleter2<ConfigOption> );
   _configOptions.clear();
   _argOptionsMap.clear();
}

//TODO: generalize?

class map_copy
{
      Config::ConfigOptionMap& dest;

   public:

      map_copy( Config::ConfigOptionMap &d ) : dest( d ) {}

      void operator()( Config::ConfigOptionMap::value_type pair ) {  dest[pair.first] = pair.second->clone(); }
};

void Config::copy ( const Config &cfg )
{
   std::for_each( cfg._configOptions.begin(), cfg._configOptions.end(), map_copy( _configOptions ) );
   for ( ConfigOptionMap::iterator it = _configOptions.begin(); it != _configOptions.end(); it++ ) {
      _argOptionsMap[it->first] = it->second;
   }
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


std::string Config::ConfigOption::getArgHelp()
{ 
   std::string help;
   std::string farg="";

// argument
   if ( _argOption != "" ) {
      farg = "--" + _argOption + " ";
      std::string argval = _option.getArgHelp();
      if ( argval.compare( "no" ) == 0 ) {
         farg += "--no-"+_argOption;
      } else {
         farg += "[=]" + argval;
      }
   } else {
      return "";
   }

// Env vars
   std::string formattedArg = "                                           ";
   formattedArg.replace( 0, farg.size(), farg );
   help += formattedArg + _message;
   return help;
}

std::string Config::ConfigOption::getEnvHelp()
{
   std::string help;
   std::string fenv="";

// Env vars
   if ( _envOption != "" ) {
      fenv = _envOption + " = " + _option.getEnvHelp();
   } else {
      return "";
   }

   std::string formattedEnv = "                                           ";
   formattedEnv.replace( 0, fenv.size(), fenv );
   help += formattedEnv + _message;
   return help;
}

