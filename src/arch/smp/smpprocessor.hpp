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

#ifndef _NANOS_SMP_PROCESSOR
#define _NANOS_SMP_PROCESSOR

#include "processingelement.hpp"
#include "config.hpp"
#include "smpthread.hpp"

//TODO: Make smp independent from pthreads? move it to OS?

namespace nanos {
namespace ext
{
   
   class SMPProcessor : public PE
   {

      private:
         // config variables
         static bool _useUserThreads;
	 static size_t _threadsStackSize;

         // disable copy constructor and assignment operator
         SMPProcessor( const SMPProcessor &pe );
         const SMPProcessor & operator= ( const SMPProcessor &pe );

      public:
         // constructors
         SMPProcessor( int id ) : PE( id,&SMP ) {}

         virtual ~SMPProcessor() {}

         virtual WD & getWorkerWD () const;
         virtual WD & getMasterWD () const;
         virtual BaseThread & createThread ( WorkDescriptor &wd );

         static void prepareConfig ( Config &config );
   };

}
}

#endif
