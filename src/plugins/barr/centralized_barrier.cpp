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

#include "barrier.hpp"
#include "system.hpp"
#include "atomic.hpp"
#include "schedule.hpp"
#include "plugin.hpp"

namespace nanos {
namespace ext {

/*! \class centralizedBarrier
    \brief implements a single semaphore barrier
*/

class centralizedBarrier: public Barrier
{

   private:
      Atomic<int> _sem;

   public:
      centralizedBarrier();
      void init();
      void barrier();
      int getSemValue() { return _sem; }
};


centralizedBarrier::centralizedBarrier(): Barrier()
{
   _sem =  0;
}

void centralizedBarrier::init() {}


void centralizedBarrier::barrier()
{
   /*! get the number of participants from the team */
   int numParticipants = myThread->getTeam()->size();

   /*! \warning We are not guaranteeing that the sem value is put back to zero at the beginning of a barrier */

   //increment the semaphore value
   _sem++;

   //wait for the semaphore value to reach numParticipants
   Scheduler::blockOnConditionLess<int>( &_sem.override(), numParticipants );

   //when it reaches that value, we increment the semaphore again
   _sem++;

   //the last thread incrementing the sem for the second time puts it at zero

   if ( _sem == ( 2*numParticipants ) ) {
      //warning: we do not have atomic assignement, thus we use atomic substraction (see atomic.hpp)
      _sem-( 2*numParticipants );
   }
}


Barrier * createCentralizedBarrier()
{
   return new centralizedBarrier();
}


/*! \class CentralizedBarrierPlugin
    \brief plugin of the related centralizedBarrier class
    \see centralizedBarrier
*/

class CentralizedBarrierPlugin : public Plugin
{

   public:
      CentralizedBarrierPlugin() : Plugin( "Centralized Barrier Plugin",1 ) {}

      virtual void init() {
         sys.setDefaultBarrFactory( createCentralizedBarrier );
      }
};

CentralizedBarrierPlugin NanosXPlugin;

}
}