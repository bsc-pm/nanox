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

#ifndef _GENERIC_EVENT_DECL
#define _GENERIC_EVENT_DECL


#include <iostream>
#include <queue>
#include <functional>

#include "workdescriptor_fwd.hpp"

#ifdef NANOS_DEBUG_ENABLED
//#define NANOS_GENERICEVENT_DEBUG
#endif

namespace nanos
{
   /****** Action definitions ******/

   /* !\brief Action virtual class
    */
   struct Action
   {
      virtual void run() = 0;
      virtual ~Action() {}
   };

   /****** Actions with static functions and wrappers ******/

   /* !\brief Action that calls a static function with no parameters
    */
   struct ActionFunPtr0 : public Action
   {
      public:
         typedef void ( *FunPtr0 )();

      private:
         FunPtr0 _fptr;

      public:
         ActionFunPtr0 ( FunPtr0 fptr ) : _fptr( fptr ) {}
         virtual void run() { _fptr(); }
   };

   /* !\brief Wrapper for static functions with no parameters
    */
   //Action* new_action( void ( *fun )() )
   Action* new_action( void ( *fun )() );


   /* !\brief Action that calls a static function with 1 parameter
    */
   template <typename T>
   struct ActionFunPtr1 : public Action
   {
      public:
         typedef void ( *FunPtr1 )( T );

      private:
         FunPtr1 _fptr;
         T _param1;

      public:
         ActionFunPtr1 ( FunPtr1 fptr, T param1 ) : _fptr( fptr ), _param1( _param1 ) {}
         virtual void run() { _fptr( _param1 ); }

   };

   /* !\brief Wrapper for static functions with 1 parameter
    */
   //template <typename Param>
   //Action* new_action( ActionFunPtr1<Param>::FunPtr1 fun, Param p );
   template <typename Param>
   Action * new_action( void ( *fun )( Param ), Param p );

   //template <typename Param>
   //Action* new_action( ActionFunPtr1::FunPtr1<Param> fun , Param p )
   //{
   //   return NEW ActionFunPtr1( fun, p );
   //}
   template <typename Param>
   Action * new_action( void ( *fun )( Param ), Param p )
   {
      return NEW ActionFunPtr1<Param>( fun, p );
   }


   /* !\brief Action that calls a static function with 3 parameters
    */
   template <typename T, typename U>
   struct ActionFunPtr2 : public Action
   {
      public:
         typedef void ( *FunPtr2 )( T, U );

      private:
         FunPtr2 _fptr;
         T _param1;
         U _param2;

      public:
         ActionFunPtr2 ( FunPtr2 fptr, T param1, U param2 ) :
            _fptr( fptr ), _param1( param1 ), _param2( param2 ) {}
         virtual void run() { _fptr( _param1, _param2 ); }

   };

   /* !\brief Wrapper for static functions with 3 parameters
    */
   template <typename Param1, typename Param2>
   Action * new_action( void ( *fun )( Param1, Param2 ), Param1 p1, Param2 p2 );

   template <typename Param1, typename Param2>
   Action * new_action( void ( *fun )( Param1, Param2 ), Param1 p1, Param2 p2 )
   {
      return NEW ActionFunPtr2<Param1, Param2>( fun, p1, p2 );
   }


   /* !\brief Action that calls a static function with 3 parameters
    */
   template <typename T, typename U, typename V>
   struct ActionFunPtr3 : public Action
   {
      public:
         typedef void ( *FunPtr3 )( T, U, V );

      private:
         FunPtr3 _fptr;
         T _param1;
         U _param2;
         V _param3;

      public:
         ActionFunPtr3 ( FunPtr3 fptr, T param1, U param2, V param3 ) :
            _fptr( fptr ), _param1( param1 ), _param2( param2 ), _param3( param3 ) {}
         virtual void run() { _fptr( _param1, _param2, _param3 ); }

   };

   /* !\brief Wrapper for static functions with 3 parameters
    */
   template <typename Param1, typename Param2, typename Param3>
   Action * new_action( void ( *fun )( Param1, Param2, Param3 ), Param1 p1, Param2 p2, Param3 p3 );

   template <typename Param1, typename Param2, typename Param3>
   Action * new_action( void ( *fun )( Param1, Param2, Param3 ), Param1 p1, Param2 p2, Param3 p3 )
   {
      return NEW ActionFunPtr3<Param1, Param2, Param3>( fun, p1, p2, p3 );
   }


   /****** Actions with member functions (non-static) and wrappers ******/

   /* !\brief Action that calls a member function with no parameters
    */
   template <typename Class>
   struct ActionMemFunPtr0 : public Action
   {
      public:
         typedef void ( Class::*MemFunPtr0 )();

      private:
         MemFunPtr0 _fptr;
         Class &_obj;

      public:
         ActionMemFunPtr0 ( MemFunPtr0 fptr, Class &obj ) : _fptr( fptr ), _obj( obj ) {}
         virtual void run() { ( _obj.*_fptr )(); }

   };

   /* !\brief Wrapper for member functions with no parameters
    */
   template <typename Class>
   Action* new_action( void ( Class::*fun )(), Class &obj );

   template <typename Class>
   Action* new_action( void ( Class::*fun )(), Class &obj )
   {
      return NEW ActionMemFunPtr0<Class>( fun, obj );
   }



   /* !\brief Action that calls a member function with 1 parameter
    */
   template <typename Class, typename T>
   struct ActionMemFunPtr1 : public Action
   {
      public:
         typedef void ( Class::*MemFunPtr1 )( T );

      private:
         MemFunPtr1 _fptr;
         Class &_obj;
         T _param1;

      public:
         ActionMemFunPtr1 ( MemFunPtr1 fptr, Class &obj, T param1 ) : _fptr( fptr ), _obj( obj ), _param1( param1 ) {}
         virtual void run() { ( _obj.*_fptr )( _param1 ); }

   };

   /* !\brief Wrapper for member functions with 1 parameter
    */
   template <typename Class, typename Param>
   Action* new_action( void ( Class::*fun )( Param ), Class &obj, Param p );

   template <typename Class, typename Param>
   Action* new_action( void ( Class::*fun )( Param ), Class &obj, Param p )
   {
      return NEW ActionMemFunPtr1<Class, Param>( fun, obj, p );
   }



   /* !\brief Action that calls a member function with no parameters, class pointer version
    */
   template <typename Class>
   struct ActionPtrMemFunPtr0 : public Action
   {
      public:
         typedef void ( Class::*PtrMemFunPtr0 )();

      private:
         PtrMemFunPtr0 _fptr;
         Class *_obj;

      public:
         ActionPtrMemFunPtr0 ( PtrMemFunPtr0 fptr, Class *obj ) : _fptr( fptr ), _obj( obj ) {}
         virtual void run() { ( _obj->*_fptr )(); }

   };

   /* !\brief Wrapper for member functions with no parameters, class pointer version
    */
   template <typename Class>
   Action* new_action( void ( Class::*fun )(), Class *obj );

   template <typename Class>
   Action* new_action( void ( Class::*fun )(), Class *obj )
   {
      return NEW ActionPtrMemFunPtr0<Class>( fun, obj );
   }



   /* !\brief Action that calls a member function with 1 parameter, class pointer version
    */
   template <typename Class, typename T>
   struct ActionPtrMemFunPtr1 : public Action
   {
      public:
         typedef void ( Class::*PtrMemFunPtr1 )( T );

      private:
         PtrMemFunPtr1 _fptr;
         Class *_obj;
         T _param1;

      public:
         ActionPtrMemFunPtr1 ( PtrMemFunPtr1 fptr, Class *obj, T param1 ) : _fptr( fptr ), _obj( obj ), _param1( param1 ) {}
         virtual void run() { ( _obj->*_fptr )( _param1 ); }

   };

   /* !\brief Wrapper for member functions with 1 parameter, class pointer version
    */
   template <typename Class, typename Param>
   Action* new_action( void ( Class::*fun )( Param ), Class *obj, Param p );

   template <typename Class, typename Param>
   Action* new_action( void ( Class::*fun )( Param ), Class *obj, Param p )
   {
      return NEW ActionPtrMemFunPtr1<Class, Param>( fun, obj, p );
   }

   /****** Actions with const member functions (non-static) and wrappers ******/

   /* !\brief Action that calls a const member function with 1 parameter
    */
   template <typename Class, typename T>
   struct ActionConstMemFunPtr1 : public Action
   {
      public:
         typedef void ( Class::*ConstMemFunPtr1 )( T ) const;

      private:
         ConstMemFunPtr1 _fptr;
         const Class &_obj;
         T _param1;

      public:
         ActionConstMemFunPtr1 ( ConstMemFunPtr1 fptr, const Class &obj, T param1 ) : _fptr( fptr ), _obj( obj ), _param1( param1 ) {}
         virtual void run() { ( _obj.*_fptr )( _param1 ); }

   };

   /* !\brief Wrapper for const member functions with 1 parameter
    */
   template <typename Class, typename Param>
   Action* new_action( void ( Class::*fun )( Param ) const, const Class &obj, Param p );

   template <typename Class, typename Param>
   Action* new_action( void ( Class::*fun )( Param ) const, const Class &obj, Param p )
   {
      return NEW ActionConstMemFunPtr1<Class, Param>( fun, obj, p );
   }



   class GenericEvent
   {
      public:
         typedef enum { CREATED, PENDING, RAISED, COMPLETED } GenericEventState;
         //typedef void * next_action_param;
         //typedef void ( *next_action_fct_0 ) ();
         //typedef std::unary_function<next_action_param, void> next_action_fct;
         //typedef void ( *next_action_fct ) ( next_action_param self );
         //typedef void ( *next_action_param ) ( void *arg );
         //typedef std::pair<next_action_fct*, next_action_param> next_action;

      protected:

         GenericEventState       _state; //! State of the event: created / pending / raised

         WD *                    _wd; //! WD related to the event

         std::queue<Action *>    _nextActions; //! Actions that must be done after event's raise

#ifdef NANOS_GENERICEVENT_DEBUG
         std::string             _description; //! Description of the event
#endif


      public:
         /*! \brief GenericEvent constructor
          */
         GenericEvent ( WD *wd ) : _state( CREATED ), _wd( wd ), _nextActions()
#ifdef NANOS_GENERICEVENT_DEBUG
         , _description()
#endif
         {}

         /*! \brief GenericEvent constructor
          */
         GenericEvent ( WD *wd, std::queue<Action *> next ) : _state( CREATED ), _wd( wd ), _nextActions( next )
#ifdef NANOS_GENERICEVENT_DEBUG
         , _description()
#endif
         {}

#ifdef NANOS_GENERICEVENT_DEBUG
         /*! \brief GenericEvent constructor
          */
         GenericEvent ( WD *wd, std::string desc ) : _state( CREATED ), _wd( wd ), _nextActions(), _description( desc ) {}

         /*! \brief GenericEvent constructor
          */
         GenericEvent ( WD *wd, std::queue<Action *> next, std::string desc ) :
            _state( CREATED ), _wd( wd ), _nextActions( next ), _description( desc ) {}
#endif

         /*! \brief GenericEvent destructor
          */
         virtual ~GenericEvent() {}


         // set/get methods
         virtual void setCreated() { _state = CREATED; }
         virtual bool isCreated() { return _state == CREATED; }

         virtual void setPending() { _state = PENDING; }
         virtual bool isPending() { return _state == PENDING; }

         virtual void setRaised() { _state = RAISED; }
         virtual bool isRaised() { return _state == RAISED; }

         virtual void setCompleted() { _state = COMPLETED; }
         virtual bool isCompleted() { return _state == COMPLETED; }

#ifdef NANOS_GENERICEVENT_DEBUG
         void setDescription( std::string desc ) { _description = desc; }
         std::string getDescription() { return _description; }
#endif

         // event related methods
         virtual void waitForEvent()
         {
            while ( _state != RAISED ) {}
         }

         WD * getWD() { return _wd; }
         void setWD( WD * wd ) { _wd = wd; }

         bool hasNextAction() { return !_nextActions.empty(); }

         Action * getNextAction()
         {
            Action * next = _nextActions.front();
            _nextActions.pop();
            return next;
         }

         void addNextAction( Action * action )
         {
            _nextActions.push( action );
         }
   };
}

#endif //_GENERIC_EVENT_DECL
