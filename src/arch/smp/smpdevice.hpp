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

#ifndef _SMP_DEVICE
#define _SMP_DEVICE

#include <stdint.h>
#include <string.h>
#include "workdescriptor_decl.hpp"

namespace nanos
{

  /* \brief Device specialization for SMP architecture
   * provides functions to allocate and copy data in the device
   */
   class SMPDevice : public Device
   {
      public:
         /*! \brief SMPDevice constructor
          */
         SMPDevice ( const char *n ) : Device ( n ) {}

         /*! \brief SMPDevice copy constructor
          */
         SMPDevice ( const SMPDevice &arch ) : Device ( arch ) {}

         /*! \brief SMPDevice destructor
          */
         ~SMPDevice() {};

        /* \breif allocate size bytes in the device
         */
         static void * allocate( size_t size )
         {
            return new char[size]; 
         }

        /* \brief free address
         */
         static void free( void *address )
         {
            delete[] (char *) address;
         }

        /* \brief copy from remoteSrc in the host to localDst in the device
         */
         static void copyIn( void *localDst, uint64_t remoteSrc, size_t size )
         {
            memcpy( localDst, (void *)remoteSrc, size );
         }

        /* \brief copy from localSrc in the device to remoteDst in the host
         */
         static void copyOut( uint64_t remoteDst, void *localSrc, size_t size )
         {
            memcpy( (void *)remoteDst, localSrc, size );
         }

        /* \brief copy localy in the device from src to dst
         */
         static void copyLocal( void *dst, void *src, size_t size )
         {
            memcpy( dst, src, size );
         }
   };
}

#endif
