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
/*
#define NORMAL       0 // -- Basis
#define ASYNC        1 // -- A little bit better (gives bad results from time to time)
#define PINNED_CUDA  0 // -- Slowdown of ~10x (gives always bad results)
#define PINNED_OS    0 // -- Similar to NORMAL (correct results though mlock fails)
#define WC           0 // -- Same as PINNED_CUDA: Slowdown of ~10x (gives always bad results)
*/


#include "workdescriptor_decl.hpp"


namespace nanos
{

typedef enum {
   NORMAL,
   ASYNC,
   PINNED_CUDA,
   PINNED_OS,
   WC
} transfer_mode;

/* \brief Device specialization for GPU architecture
 * provides functions to allocate and copy data in the device
 */

   class GPUDevice : public Device
   {
      private:
         static transfer_mode _transferMode;

         static unsigned int _rlimit;

         static void getMemoryLockLimit();

      public:
         /*! \brief GPUDevice constructor
          */
         GPUDevice ( const char *n ) : Device ( n )
         {
            getMemoryLockLimit();
         }

         /*! \brief GPUDevice copy constructor
          */
         GPUDevice ( const GPUDevice &arch ) : Device ( arch ) {}

         /*! \brief GPUDevice destructor
          */
         ~GPUDevice() {};

         static void setTransferMode ( transfer_mode mode )
         {
            _transferMode = mode;
         }

         static transfer_mode getTransferMode ()
         {
            return _transferMode;
         }

         static void * allocateWholeMemory( size_t size );

         static void * allocate( size_t size );
         static void free( void *address );

         static void copyIn( void *localDst, uint64_t remoteSrc, size_t size );
         static void copyOut( uint64_t remoteDst, void *localSrc, size_t size );

         static void copyLocal( void *dst, void *src, size_t size );

         static void copyOutAsyncToBuffer( void * src, void * dst, size_t size );
         static void copyOutAsyncWait();
         static void copyOutAsyncToHost( void * src, void * dst, size_t size );
         static void copyOutSyncToHost ( void * dst, void * src, size_t size );


   };
}

#endif
