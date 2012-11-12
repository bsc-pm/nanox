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

#ifndef _NANOS_MPI_WD
#define _NANOS_MPI_WD

#include <stdint.h>
#include "mpidevice.hpp"
#include "workdescriptor.hpp"
#include "config.hpp"

namespace nanos {
namespace ext
{

   extern MPIDevice MPI;
   
   class MPIDD : public DD
   {

      public:
         typedef void ( *work_fct ) ( void *self );

      private:
         work_fct       _work;
         intptr_t *     _stack;
         intptr_t *     _state;
         static size_t     _stackSize;

      public:
         // constructors
         MPIDD( work_fct w ) : DD( &MPI ),_work( w ),_stack( 0 ),_state( 0 ) {}

         MPIDD() : DD( &MPI ),_work( 0 ),_stack( 0 ),_state( 0 ) {}

         // copy constructors
         MPIDD( const MPIDD &dd ) : DD( dd ), _work( dd._work ), _stack( 0 ), _state( 0 ) {}

         // assignment operator
         const MPIDD & operator= ( const MPIDD &wd );
         // destructor

         virtual ~MPIDD() { if ( _stack ) delete[] _stack; }

         work_fct getWorkFct() const { return _work; }

         bool hasStack() { return _state != NULL; }

         void initStack( void *data );

        /* \brief Wrapper called by the instrumented library to
         * be able to instrument the exact moment in which the runtime
         * is left and the user's code starts being executed.
         */
         static void workWrapper( void *data );

         intptr_t *getState() const { return _state; }

         void setState ( intptr_t * newState ) { _state = newState; }

         static void prepareConfig( Config &config );

         virtual void lazyInit (WD &wd, bool isUserLevelThread, WD *previous);
         virtual size_t size ( void ) { return sizeof(MPIDD); }
         virtual MPIDD *copyTo ( void *toAddr );

         virtual MPIDD *clone () const { return NEW MPIDD ( *this); }
      };

   inline const MPIDD & MPIDD::operator= ( const MPIDD &dd )
   {
      // self-assignment: ok
      if ( &dd == this ) return *this;

      DD::operator= ( dd );

      _work = dd._work;
      _stack = 0;
      _state = 0;

      return *this;
   }

}
}

#endif
