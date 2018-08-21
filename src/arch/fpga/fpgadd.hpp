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

#ifndef _NANOS_FPGA_DD
#define _NANOS_FPGA_DD

#include "fpgadevice.hpp"
#include "workdescriptor.hpp"

namespace nanos {
namespace ext {

      extern FPGADevice FPGA;
      class FPGADD : public DD
      {

         private:
            int            _accNum; //! Accelerator that will run the task

         public:
            // constructors
            FPGADD( work_fct w , int accNum ) : DD( &FPGA, w ), _accNum( accNum ) {}

            FPGADD() : DD( &FPGA, NULL ) {}

            // copy constructors
            FPGADD( const FPGADD &dd ) : DD( dd ), _accNum( dd._accNum ){}

            // assignment operator
            const FPGADD & operator= ( const FPGADD &wd );

            // destructor
            virtual ~FPGADD() { }

            virtual void lazyInit ( WD &wd, bool isUserLevelThread, WD *previous ) { }
            virtual size_t size ( void ) { return sizeof( FPGADD ); }
            virtual FPGADD *copyTo ( void *toAddr );
            virtual FPGADD *clone () const { return NEW FPGADD ( *this ); }

            virtual bool isCompatibleWithPE ( const ProcessingElement *pe=NULL );
      };
      inline const FPGADD & FPGADD::operator= ( const FPGADD &dd )
      {
         // self-assignment: ok
         if ( &dd == this ) return *this;

         DD::operator= ( dd );

         return *this;
      }


} // namespace ext
} // namespace nanos

#endif

