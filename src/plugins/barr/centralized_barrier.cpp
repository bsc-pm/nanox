#include "barrier.hpp"
#include "system.hpp"
#include "atomic.hpp"
#include "schedule.hpp"
#include "plugin.hpp"

using namespace nanos;

/*! \class centralizedBarrier
    \brief implements a single semaphore barrier
*/

class centralizedBarrier: public Barrier
{

   private:
      Atomic<int> sem;

   public:
      centralizedBarrier();
      void init();
      void barrier();
      int getSemValue() { return sem; }
};


centralizedBarrier::centralizedBarrier(): Barrier()
{
   sem =  0;
}

void centralizedBarrier::init() {}


void centralizedBarrier::barrier()
{
   /*! get the number of participants from the team */
   numParticipants = myThread->getTeam()->size();

   /*! \warning We are not guaranteeing that the sem value is put back to zero at the beginning of a barrier */

   //increment the semaphore value
   sem++;

   //wait for the semaphore value to reach numParticipants
   Scheduler::blockOnConditionLess<int>( &sem.override(), numParticipants );

   //when it reaches that value, we increment the semaphore again
   sem++;

   //the last thread incrementing the sem for the second time puts it at zero

   if ( sem == ( 2*numParticipants ) ) {
      //warning: we do not have atomic assignement, thus we use atomic substraction (see atomic.hpp)
      sem-( 2*numParticipants );
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
