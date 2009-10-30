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

#include <assert.h>
#include <pthread.h>

#include "barrier.hpp"
#include "system.hpp"
#include "atomic.hpp"
#include "plugin.hpp"


using namespace nanos;

/*! \class posixBarrier
    \brief implements a barrier according to a centralized scheme with a posix barrier
*/

class posixBarrier: public Barrier
{

   private:
      pthread_barrier_t pBarrier;

   public:
      /*! \warning the creation of the pthread_barrier_t variable will be performed when the barrier function is invoked
                   because only at that time we exectly know the number of participants (which is dynamic, as in a team
                   threads can dynamically enter and exit)
      */
      posixBarrier() { }

      void init() { }

      void barrier();
};


void posixBarrier::barrier()
{
   /*! get the number of participants from the team */
   int numParticipants = myThread->getTeam()->size();

   /*! initialize the barrier to the current participant number */
   pthread_barrier_init ( &pBarrier, NULL, numParticipants );

   pthread_barrier_wait( &pBarrier );
}


Barrier * createPosixBarrier()
{
   return new posixBarrier();
}


/*! \class PosixBarrierPlugin
    \brief plugin of the related posixBarrier class
    \see posixBarrier
*/

class PosixBarrierPlugin : public Plugin
{

   public:
      PosixBarrierPlugin() : Plugin( "Posix Barrier Plugin",1 ) {}

      virtual void init() {
         sys.setDefaultBarrFactory( createPosixBarrier );
      }
};

PosixBarrierPlugin NanosXPlugin;

