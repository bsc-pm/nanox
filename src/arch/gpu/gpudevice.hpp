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

#ifndef _GPU_DEVICE
#define _GPU_DEVICE

#define NORMAL       1 // -- Basis
#define ASYNC        0 // -- A little bit better (gives bad results from time to time)
#define PINNED_CUDA  0 // -- Slowdown of ~10x (gives always bad results)
#define PINNED_OS    0 // -- Similar to NORMAL (correct results though mlock fails)
#define WC           0 // -- Same as PINNED_CUDA: Slowdown of ~10x (gives always bad results)



#include "workdescriptor_decl.hpp"
#if ASYNC | PINNED_CUDA | WC
#include <map>
#endif


namespace nanos
{

/* \brief Device specialization for GPU architecture
 * provides functions to allocate and copy data in the device
 */

   class GPUDevice : public Device
   {
      private:
#if ASYNC | PINNED_CUDA | WC
         static std::map< void *, uint64_t > _pinnedMemory;
#endif

      public:
         /*! \brief GPUDevice constructor
          */
         GPUDevice ( const char *n ) : Device ( n ) {}

         /*! \brief GPUDevice copy constructor
          */
         GPUDevice ( const GPUDevice &arch ) : Device ( arch ) {}

         /*! \brief GPUDevice destructor
          */
         ~GPUDevice() {};

         static void * allocate( size_t size );
         static void free( void *address );

         static void copyIn( void *localDst, uint64_t remoteSrc, size_t size );
         static void copyOut( uint64_t remoteDst, void *localSrc, size_t size );

         static void copyLocal( void *dst, void *src, size_t size );

   };
}

#endif
