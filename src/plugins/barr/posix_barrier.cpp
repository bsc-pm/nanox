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
   numParticipants = myThread->getTeam()->size();

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

