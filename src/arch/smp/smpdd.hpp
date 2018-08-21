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

#ifndef _NANOS_SMP_WD
#define _NANOS_SMP_WD

#include <stdint.h>
#include "smpdevice_decl.hpp"
#include "workdescriptor_fwd.hpp"
#include "config.hpp"

namespace nanos {
namespace ext {

   //extern SMPDevice SMP;
   SMPDevice &getSMPDevice();

   //! \brief Device Data for SMP
   class SMPDD : public DD
   {
      private:
         void               *_stack;             //!< Stack base
         void               *_state;             //!< Stack pointer
         static size_t       _stackSize;         //!< Stack size
      protected:
         SMPDD( work_fct w, Device *dd ) : DD( dd, w ),_stack( 0 ),_state( 0 ) {}
         SMPDD( Device *dd ) : DD( dd, NULL ), _stack( 0 ),_state( 0 ) {}
      public:
         //! \brief Constructor using work function
         SMPDD( work_fct w ) : DD( &getSMPDevice(), w ), _stack( 0 ),_state( 0 ) {}
         //! \brief Default constructor 
         SMPDD() : DD( &getSMPDevice(), NULL ), _stack( 0 ),_state( 0 ) {}
         //! \brief Copy constructor
         SMPDD( const SMPDD &dd ) : DD( dd ), _stack( 0 ), _state( 0 ) {}
         //! \brief Assignment operator
         const SMPDD & operator= ( const SMPDD &wd );
         //! \brief Destructor
         virtual ~SMPDD() { if ( _stack ) delete[] (char *) _stack; }

         bool hasStack() { return _state != NULL; }

         void initStack( WD *wd );

        /*! \brief Wrapper called to be able to instrument the
         * exact moment in which the runtime is left and the
         * user's code starts being executed and to be able to
         * re-execute it (fault tolerance).
         */
         static void workWrapper( WD &data );

         //! \brief Getting current stack pointer
         void * getState() const { return _state; }
         //! \brief Setting current stack pointer
         void setState ( void * newState ) { _state = newState; }

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

} // namespace ext
} // namespace nanos

#endif
