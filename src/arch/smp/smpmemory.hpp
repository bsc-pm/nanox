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

#ifndef _SMP_COPIER
#define _SMP_COPIER

namespace nanos
{

   class SMPMemory
   {
      public:
         void * allocate( size_t size )
         {
            return new char[size]; 
         }

         void free( void *address )
         {
            delete[] (char *) address;
         }

         void copyIn( void *localDst, void* remoteSrc, size_t size )
         {
            memcpy( localDst, remoteSrc, size );
         }

         void copyOut( void *remoteDst, void *localSrc, size_t size )
         {
            memcpy( remoteDst, localSrc, size );
         }

         void copyLocal( void *dst, void * src, size_t size )
         {
            memcpy( dst, src, size );
         }
   };
}

#endif
