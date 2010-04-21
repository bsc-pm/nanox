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

#ifndef _NANOS_GPU_WD
#define _NANOS_GPU_WD

#include "config.hpp"
#include "gpudevice.hpp"
#include "workdescriptor.hpp"

namespace nanos {
namespace ext
{

   extern GPUDevice GPU;

   class GPUDD : public DD
   {

      public:
         typedef void ( *work_fct ) ( void *self );

      private:
         static int     _gpuCount; // Number of CUDA-capable GPUs
         work_fct       _work;
         intptr_t *     _state;

      public:
         // constructors
         GPUDD( work_fct w ) : DD( &GPU ), _work( w ), _state( 0 ) {}

         GPUDD() : DD( &GPU ), _work( 0 ), _state( 0 ) {}

         // copy constructors
         GPUDD( const GPUDD &dd ) : DD( dd ), _work( dd._work ), _state( 0 ) {}

         // assignment operator
         const GPUDD & operator= ( const GPUDD &wd );

         // destructor
         virtual ~GPUDD() { }

         work_fct getWorkFct() const { return _work; }

         bool hasStack() { return false; }

         intptr_t *getState() const { return _state; }

         void setState ( intptr_t * newState ) { _state = newState; }

         static void prepareConfig( Config &config );

         static int getGPUCount () { return _gpuCount; }

         virtual void lazyInit (WD &wd, bool isUserLevelThread, WD *previous) { }
         virtual size_t size ( void ) { return sizeof(GPUDD); }
         virtual GPUDD *copyTo ( void *toAddr );
      };

   inline const GPUDD & GPUDD::operator= ( const GPUDD &dd )
   {
      // self-assignment: ok
      if ( &dd == this ) return *this;

      DD::operator= ( dd );

      _gpuCount = dd._gpuCount;
      _work = dd._work;
      _state = 0;

      return *this;
   }

}
}

#endif
