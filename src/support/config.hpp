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

#ifndef _NANOS_CONFIG
#define _NANOS_CONFIG

#include <debug.hpp>
#include <stdexcept>
#include <vector>
#include "config_decl.hpp"
#include <memory>
#include <sstream>
#include <string.h>

namespace nanos {

template<typename T>
inline bool Config::CheckValue<T>::operator() ( T &value, char suffix ) const
{
   return true;
}

template<typename T>
inline bool Config::isPositive<T>::operator() ( T &value, char suffix ) const
{
   return value > 0;
}

template<typename T>
inline bool Config::isMetric<T>::operator() ( T &value, char suffix ) const
{
   bool rv = false;
   switch (suffix) {
      case  0 :
      case 'B':
      case 'b':
         rv = true;
         break;
      case 'K':
      case 'k':
         value = value << 10;
         rv = true;
         break;
      case 'M':
      case 'm':
         value = value << 20;
         rv = true;
         break;
      case 'G':
      case 'g':
         value = value << 30;
         rv = true;
         break;
      default:
         value = 0;
         break;
   }

   return rv;
}

inline std::string Config::getOrphanOptions()
{
   ensure0( _orphanOptionsMap != NULL, "Config::_orphanOptionsMap was not initialised" );

   std::string str;
   bool first = true;

   for ( ConfigOrphansMap::const_iterator it = _orphanOptionsMap->begin();
      it != _orphanOptionsMap->end(); ++it )
   {
      // If the argument has not been parsed
      if ( it->second == false ) {
         // Append separator if it's not the first element
         if ( !first )
            str += ", ";
         else
            first = false;

         str += it->first;

      }
   }

   return str;
}

inline const Config::Option & Config::Option::operator= ( const Config::Option &opt )
{
   // self-assigment: ok
   this->_name = opt._name;
   this->_type = opt._type;
   return *this;
}

inline const std::string & Config::Option::getName() const
{
   return _name;
}

inline void  Config::Option::setName( const std::string &name )
{
   _name = name;
}

inline const Config::Option::OptType& Config::Option::getType() const
{
   return _type;
}

template<typename T, class helpFormat, typename checkT>
const Config::ActionOption<T,helpFormat,checkT> &
Config::ActionOption<T,helpFormat,checkT>::operator=
( const Config::ActionOption<T,helpFormat,checkT> & opt )
{
   // self-assigment: ok
   Option::operator=( opt );
   this->_check = opt._check;
   return *this;
}

template<typename T, class helpFormat, typename checkT>
void Config::ActionOption<T,helpFormat,checkT>::parse ( const char *value )
{
   T t;
   char suffix = 0;
   std::istringstream iss( value );

   // taking advantage of the fact that istream::operator>> skips whitespaces
   if ( ( iss >> t ).fail() )
      throw InvalidOptionException( *this,value );

   if ( (!iss.eof()) && ( ( iss >> suffix ).fail() ) )
      throw InvalidOptionException( *this,value );

   if ( ! checkValue( t,suffix ) )
      throw InvalidOptionException( *this,value );

   setValue( t );
}

template<typename T, class helpFormat, typename checkT>
inline bool Config::ActionOption<T,helpFormat,checkT>::checkValue ( T &value, char suffix ) const
{
   return _check( value, suffix );
}

template<typename T, class helpFormat, typename checkT>
inline const std::string Config::ActionOption<T,helpFormat,checkT>::getArgHelp ()
{
   return std::string("<" + _helpFormat() + ">");
}

template<typename T, class helpFormat, typename checkT>
inline const std::string Config::ActionOption<T,helpFormat,checkT>::getEnvHelp ()
{
   return std::string("<" + _helpFormat() + ">");
}

template<typename T, class helpFormat, typename checkT>
inline void Config::VarOption<T,helpFormat,checkT>::setValue ( const T &value )
{
   _var = value;
}

template<typename T, class helpFormat, typename checkT>
inline Config::VarOption<T,helpFormat,checkT> * Config::VarOption<T,helpFormat,checkT>::clone ()
{
   return NEW VarOption( *this );
}


// FIXME: new list
template<typename T, class helpFormat, typename checkT>
inline void Config::ListOption<T,helpFormat,checkT>::setValue ( const T &value )
{
   _var.push_back( value );
}

template<typename T, class helpFormat, typename checkT>
inline Config::ListOption<T,helpFormat,checkT> * Config::ListOption<T,helpFormat,checkT>::clone ()
{
   return NEW ListOption( *this );
}

template<typename T, class helpFormat, typename checkT>
void Config::ListOption<T,helpFormat,checkT>::parse ( const char *value )
{
   std::istringstream input( value );
   // taking advantage of the fact that istream::operator>> skips whitespaces
   for (std::string element; std::getline(input, element, _sep); ) {
      std::stringstream iss(element);
      T t;
      char suffix = 0;
      if ( ( iss >> t ).fail() )
         throw InvalidOptionException( *this, value );
      if ( (!iss.eof()) && ( iss>>suffix ).fail() )
         throw InvalidOptionException( *this, value );
      if ( ! this->checkValue( t,suffix ) )
         throw InvalidOptionException( *this, value );
      setValue( t );
   }
}
// FIXME: end new list

inline std::string Config::HelpFormat::operator()()
{
   return "value";
}

inline std::string Config::IntegerHelpFormat::operator()()
{
   return "integer";
}

inline std::string Config::MetricHelpFormat::operator()()
{
   return "integer + suffix";
}

inline std::string Config::BoolHelpFormat::operator()()
{
   return "true/false";
}

inline std::string Config::StringHelpFormat::operator()()
{
   return "string";
}

inline std::string Config::PositiveHelpFormat::operator()()
{
   return "positive integer";
}

template<typename T, class helpFormat, typename checkT>
inline void Config::FuncOption<T,helpFormat,checkT>::setValue ( const T& value )
{
   _function(value);
}

template<typename T, class helpFormat, typename checkT>
inline Config::FuncOption<T,helpFormat,checkT> * Config::FuncOption<T,helpFormat,checkT>::clone ()
{
   return NEW FuncOption( *this );
}

template<typename T>
inline const std::string Config::MapAction<T>::getHelp ()
{
   std::string help = "";
   for ( unsigned int i = 0; i < _options.size(); i++ ) {
      help += ( (i == 0) ? "": ", ") + _options[i].first;
   }
   return help;
}

template<typename T>
inline Config::MapAction<T>::MapAction( ) : Option( Option::VALUE ), _options()
{
   _options.reserve(16);
}

template<typename T>
const Config::MapAction<T> & Config::MapAction<T>::operator= ( const Config::MapAction<T> &opt )
{
   // self->assigment: ok
   Option::operator=( opt );
   this->_options = opt._options;
   return *this;
}

template<typename T>
void Config::MapAction<T>::parse ( const char *value )
{
   typename MapList::const_iterator it;

   for ( it = _options.begin(); it < _options.end(); it++ ) {
      if ( value == it->first ) {
         setValue( it->second );
         return;
      }
   }

   throw InvalidOptionException( *this,value );
}

template<typename T>
inline Config::MapAction<T> & Config::MapAction<T>::addOption ( std::string optionName, T value )
{
   _options.push_back( MapOption( optionName, value ) );
   return *this;
}

template<typename T>
inline const std::string Config::MapAction<T>::getArgHelp ()
{
   return getHelp();
}

template<typename T>
inline const std::string Config::MapAction<T>::getEnvHelp ()
{
   return getHelp();
}

template <typename T>
inline void Config::MapVar<T>::setValue ( const T &value ) { _var = value; }

template <typename T>
inline Config::MapVar<T> * Config::MapVar<T>::clone () { return NEW MapVar( *this ); }

inline Config::PluginVar & Config::PluginVar::addOption ( const std::string & value )
{
   MapVar<std::string>::addOption( value, value );
   return *this;
}

inline void Config::ActionFlag::parse ( const char *value )
{
   if ( strcasecmp(value, "yes" ) == 0) {
      setValue( true );
   } else if ( strcasecmp(value, "no" ) == 0 ) {
      setValue( false );
   } else
      throw InvalidOptionException( *this,value );
}

inline const std::string Config::ActionFlag::getArgHelp()
{
   return "no";
}

inline const std::string Config::ActionFlag::getEnvHelp()
{
   return "yes/no";
}

inline void Config::FlagOption::setValue ( const bool &value )
{
   _var = !( value ^ _setTo );
}

inline Config::FlagOption * Config::FlagOption::clone ()
{
   // We cannot use Memtracker NEW's macro
   return new FlagOption( *this );
}

inline const Config::HelpTriplet& Config::HelpTriplet::operator=( const HelpTriplet& ht )
{
   this->_envHelp = ht._envHelp;
   this->_argHelp = ht._argHelp;
   this->_message = ht._message;
   return *this;
}

inline const std::string& Config::BaseConfigOption::getEnvVar()
{
   return _envOption;
}

inline const std::string& Config::BaseConfigOption::getArg()
{
   return _argOption;
}

inline void Config::BaseConfigOption::setEnvVar( const std::string envOption )
{
   _envOption = envOption;
}

inline void Config::BaseConfigOption::setArg( const std::string argOption )
{
   _argOption = argOption;
}

inline Config::Option& Config::BaseConfigOption::getOption()
{
   return _option;
}

inline const std::string Config::BaseConfigOption::getSection ()
{
   return _section;
}

inline InvalidOptionException::InvalidOptionException( const Config::Option &option, const std::string &value ) :
               runtime_error( std::string( "Ignoring invalid value '" )+value +"' for "+option.getName() ) {}

inline const Config::BaseConfigOption& Config::BaseConfigOption::operator= ( const BaseConfigOption &co )
{
   if ( this == &co )
      return *this;
   this->_optionName = co._optionName;
   this->_envOption = co._envOption;
   this->_argOption = _argOption;
   this->_option = _option;
   this->_message = _message;
   this->_section = _section;
   return *this;
}

inline Config::BaseConfigOption* Config::ConfigOption::clone()
{
   // We cannot use Memtracker NEW's macro
   return new ConfigOption( _optionName, _envOption, _argOption, *(_option.clone()), _message, _section);
}

inline Config::BaseConfigOption* Config::ConfigAliasOption::clone()
{
   // We cannot use Memtracker NEW's macro
   return new ConfigAliasOption( _optionName, _envOption, _argOption, *(_option.clone()), _message, _section);
}

} // namespace nanos

#endif
