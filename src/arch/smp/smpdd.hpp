/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

#ifndef _NANOS_SMP_WD
#define _NANOS_SMP_WD

#include <stdint.h>
#include "smpdevice.hpp"
#include "workdescriptor_fwd.hpp"
#include "config.hpp"

namespace nanos {
namespace ext
{

   extern SMPDevice SMP;

   class SMPDD : public DD
   {
      private:
         intptr_t *     _stack;
         intptr_t *     _state;
         static size_t     _stackSize;

      protected:
         SMPDD( work_fct w, Device *dd ) : DD( dd, w ),_stack( 0 ),_state( 0 ) {}
         SMPDD( Device *dd ) : DD( dd, NULL ), _stack( 0 ),_state( 0 ) {}

      public:
         // constructors
         SMPDD( work_fct w ) : DD( &SMP, w ), _stack( 0 ),_state( 0 ) {}

         SMPDD() : DD( &SMP, NULL ), _stack( 0 ),_state( 0 ) {}

         // copy constructors
         SMPDD( const SMPDD &dd ) : DD( dd ), _stack( 0 ), _state( 0 ) {}

         // assignment operator
         const SMPDD & operator= ( const SMPDD &wd );
         // destructor

         virtual ~SMPDD() { if ( _stack ) delete[] _stack; }

         bool hasStack() { return _state != NULL; }

         void initStack( WD *wd );

        /*! \brief Wrapper called to be able to instrument the
         * exact moment in which the runtime is left and the
         * user's code starts being executed and to be able to
         * re-execute it (fault tolerance).
         */
         static void workWrapper( WD &data );

         intptr_t *getState() const { return _state; }

         void setState ( intptr_t * newState ) { _state = newState; }

         static void prepareConfig( Config &config );

         virtual void lazyInit (WD &wd, bool isUserLevelThread, WD *previous);
         virtual size_t size ( void ) { return sizeof(SMPDD); }
         virtual SMPDD *copyTo ( void *toAddr );

         virtual SMPDD *clone () const { return NEW SMPDD ( *this); }

            /*! \brief Encapsulates the user function call.
             * This avoids code duplication for additional
             * operations that must be done just before/after
             * this call (e.g. task re-execution on errors).
             */
            void execute ( WD &wd ) throw();

#ifdef NANOS_RESILIENCY_ENABLED
            /*! \brief Restores the workdescriptor to its original state.
             * Leaving the recovery dependent to the arch allows more
             * accurate recovery for each kind of device.
             */
            void recover ( WD &wd );
#endif
   };

   inline const SMPDD & SMPDD::operator= ( const SMPDD &dd )
   {
      // self-assignment: ok
      if ( &dd == this ) return *this;

      DD::operator= ( dd );

      _stack = 0;
      _state = 0;

      return *this;
   }

}
}

#endif
