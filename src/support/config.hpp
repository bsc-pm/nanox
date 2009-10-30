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

#ifndef _NANOS_CONFIG
#define _NANOS_CONFIG

#include <stdexcept>
#include <vector>
#include "compatibility.hpp"
#include <memory>
#include <sstream>

namespace nanos
{

   class Config
   {

      public:
         // Checking predicates
         // true predicate

         template<typename T> class CheckValue
         {

            public:
               virtual ~CheckValue() {}

               virtual bool operator() ( const T &value ) const { return true; };
         };

         // isPositive predicate

         template<typename T> class isPositive : public CheckValue<T>
         {

            public:
               virtual ~isPositive() {}

               virtual bool operator() ( const T &value ) const { return value > 0; }
         };

         /** Configuration options */
         // Abstract Base Class

         class Option
         {

            public:
               typedef enum { FLAG, VALUE } OptType;

            private:
               std::string name;
               OptType type;

            public:
               // constructors
               Option( const std::string &n, const OptType t ) : name( n ), type( t ) {}

               Option( const char *n, const OptType t ) : name( n ), type( t ) {}

               // copy constructor
               Option( const Option &opt ) : name( opt.name ),type( opt.type ) {}

               // assignment operator
               const Option & operator= ( const Option &opt );
               // destructors
               virtual ~Option() {};

               const std::string &getName() const { return name; }

               const OptType& getType() const { return type; }

               virtual void parse ( const char *value ) = 0;

               // clone idiom
               virtual Option * clone () = 0;
         };

         // Action Option, Base Class
         template<typename T, typename checkT = CheckValue<T> >

         class ActionOption : public Option
         {

            private:
               checkT check;

            public:
               // constructors
               ActionOption( const std::string &name ) :
                     Option( name,Option::VALUE ) {}

               ActionOption( const char *name ) :
                     Option( name,Option::VALUE ) {}

               // copy constructors
               ActionOption( const ActionOption& opt ) : Option( opt ), check( opt.check ) {}

               // assignment operator
               const ActionOption & operator= ( const ActionOption & opt );
               // destructor
               virtual ~ActionOption() {}

               virtual void parse ( const char *value );
               virtual void setValue ( const T &value ) = 0;

               bool checkValue ( const T &value ) const	{ return check( value );  };

               virtual ActionOption * clone () = 0;
         };

         // VarOption: Option modifies a variable
         template<typename T, typename checkT= CheckValue<T> >

         class VarOption : public ActionOption<T,checkT>
         {

            private:
               T &var;
               // assignment operator
               const VarOption & operator= ( const VarOption &opt );

            public:
               //constructors
               VarOption( const std::string &name,T &ref ) :
                     ActionOption<T,checkT>( name ),var( ref ) {}

               VarOption( const char *name, T &ref ) :
                     ActionOption<T,checkT>( name ),var( ref ) {}

               // copy constructor
               VarOption( const VarOption &opt ) :
                     ActionOption<T,checkT>( opt ),var( opt.var ) {}

               //destructor
               virtual ~VarOption() {}

               virtual void setValue ( const T &value ) { var = value; };

               virtual VarOption * clone () { return new VarOption( *this ); };
         };

         // shortcuts for VarOptions and ActionOptions

         typedef class VarOption<int> 				IntegerVar;

         typedef class VarOption<bool> 				BoolVar;

         typedef class VarOption<std::string> 			StringVar;

         typedef class VarOption<int,isPositive<int> > 		PositiveVar;

         typedef class ActionOption<int> 			IntegerAction;

         typedef class ActionOption<bool> 			BoolAction;

         typedef class ActionOption<std::string> 		StringAction;

         typedef class ActionOption<int,isPositive<int> > 	PositiveAction;

         template<typename T> class MapAction : public Option
         {

            public:
               typedef std::pair<std::string,T> MapOption;
               typedef std::vector<MapOption> MapList;

            private:
               const MapList options;

            public:
               // constructors
               MapAction( const std::string &name, const MapList &opts ) :
                     Option( name,Option::VALUE ), options( opts ) {}

               MapAction( const char *name, const MapList &opts ) :
                     Option( name,Option::VALUE ), options( opts ) {}

               // copy constructor
               MapAction( const MapAction &opt ) : Option( opt ),options( opt.options ) {}

               // assignment operator
               const MapAction & operator= ( const MapAction &opt );
               // destructor
               virtual ~MapAction() {}

               virtual void parse ( const char *name );
               virtual void setValue ( const T &value ) = 0;
               virtual MapAction * clone () = 0;
         };

         template<typename T> class MapVar : public MapAction<T>
         {

            private:
               T &var;
               // assignment operator
               const MapVar & operator= ( const MapVar & );

            public:
               typedef typename MapAction<T>::MapOption MapOption;
               typedef typename MapAction<T>::MapList MapList;

               //constructors
               MapVar( const std::string &name, T &ref, const MapList &opts ) :
                     MapAction<T>( name,opts ), var( ref ) {}

               MapVar( const char *name, T &ref, const MapList &opts ) :
                     MapAction<T>( name,opts ), var( ref ) {}

               // copy constructor
               MapVar( const MapVar &opt ) : MapAction<T>( opt ), var( opt.var ) {}

               // destructor
               virtual ~MapVar() {}

               virtual void setValue ( const T &value ) { var = value; };

               virtual MapVar * clone () { return new MapVar( *this ); };
         };

         // TODO: make final class
         // TODO: inverted flags?
         // TODO: add action in same pattern as other options

         class FlagOption : public Option
         {

            private:
               bool &var;
               bool  setTo;
               // assigment operator
               const FlagOption & operator= ( const FlagOption &opt );

            public:
               // constructors
               FlagOption ( const std::string &name, bool &ref, bool value=true ) :
                     Option( name,Option::FLAG ),var( ref ),setTo( value ) {}

               FlagOption ( const char *name, bool &ref, bool value=true ) :
                     Option( name,Option::FLAG ),var( ref ),setTo( value ) {}

               // copy constructors
               FlagOption( const FlagOption &opt ) : Option( opt ), var( opt.var ), setTo( opt.setTo ) {}

               // destructor
               virtual ~FlagOption() {}

               virtual void parse ( const char *value );
               virtual FlagOption * clone () { return new FlagOption( *this ); };
         };

         typedef TR1::unordered_map<std::string, Option *> OptionMap;
         typedef std::vector<Option *> OptionList;

      private:
         OptionList envOptions;
         OptionMap  argOptions;

      protected:

         virtual void setDefaults();
         void parseFiles();
         void parseArguments();
         void parseEnvironment();
         void clear();
         void copy( const Config &origin );

      public:
         // constructors
         Config() {}

         // copy constructors
         Config( const Config &cfg );
         // assignment operator
         const Config & operator= ( const Config &cfg );
         // destructor
         virtual ~Config ();

         void init();
         void registerEnvOption ( Option *opt );
         void registerArgOption ( Option *opt );
   };

   /** exceptions */

   class InvalidOptionException : public  std::runtime_error
   {

      public:
         InvalidOptionException( const Config::Option &option,
                                 const std::string &value ) :
               runtime_error( std::string( "Ignoring invalid value '" )+value
                              +"' for "+option.getName() ) {}

   };

   /** inline functions */

   inline const Config::Option & Config::Option::operator= ( const Config::Option &opt )
   {
      // self-assigment: ok
      this->name = opt.name;
      this->type = opt.type;
      return *this;
   }

   template<typename T,typename checkT>
   const Config::ActionOption<T,checkT> &
   Config::ActionOption<T,checkT>::operator=
   ( const Config::ActionOption<T,checkT> & opt )
   {
      // self-assigment: ok
      Option::operator=( opt );
      this->check = opt.check;
      return *this;
   }

   template<typename T>
   const Config::MapAction<T> & Config::MapAction<T>::operator= ( const Config::MapAction<T> &opt )
   {
      // self->assigment: ok
      Option::operator=( opt );
      this->options = opt.options;
      return *this;
   }

   template<typename T,typename checkT>
   void Config::ActionOption<T,checkT>::parse ( const char *value )
   {
      T t;
      std::istringstream iss( value );

      if ( ( iss >> t ).fail() )
         throw InvalidOptionException( *this,value );

      // TODO: check for remaining chars
      if ( ! checkValue( t ) )
         throw InvalidOptionException( *this,value );

      setValue( t );
   }

   template<typename T>
   void Config::MapAction<T>::parse ( const char *value )
   {
      typename MapList::const_iterator it;

      for ( it = options.begin(); it < options.end(); it++ ) {
         if ( value == it->first ) {
            setValue( it->second );
            return;
         }
      }

      throw InvalidOptionException( *this,value );
   }

   inline void Config::FlagOption::parse ( const char *value )
   {
      if ( value )
         throw InvalidOptionException( *this,value );

      var = setTo;
   }


};

#endif
