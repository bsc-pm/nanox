#include <assert.h>
#include <vector>
#include <math.h>

#include "barrier.hpp"
#include "system.hpp"
#include "atomic.hpp"
#include "plugin.hpp"


using namespace nanos;

using namespace std;


/*!
   \class disseminationBarrier
   \brief implements a barrier according to the dissemination algorithm

 */

class disseminationBarrier: public Barrier
{

   private:
      /*! the semaphores are implemented with a vector of atomic integers, because their number can change in successive
          invocations of the barrier method */
      vector<Atomic<int> > semaphores;

      /*! q is the log of threadNum */
      int q;

   public:
      /*! \warning the creation of the pthread_barrier_t variable will be performed when the barrier function is invoked
                   because only at that time we exectly know the number of participants (which is dynamic, as in a team
                   threads can dynamically enter and exit)
      */
      disseminationBarrier() { }

      void init() { }

      void barrier();
      void barrier( int id );
};


void disseminationBarrier::barrier()
{
   int myID = -1;

   for ( int s = 0; s < q; s++ ) {
      int toSign = ( myID + ( int ) pow( 2,s ) ) % numParticipants;
      ( semaphores[toSign] )--;

      while ( semaphores[myID] != 0 ) {}

      if ( s < q ) {
         //reset the semaphore for the next round
         semaphores[myID]++; //notice that we allow another thread to signal it before we reset it (not set to 1!)
      }
   }

   //reset the semaphore for the next barrier
   //warning: as we do not have an atomic assignement, we use the substraction operation
   semaphores[myID]-1;
}


void disseminationBarrier::barrier( int myID )
{
   /*! get the number of participants from the team */
   numParticipants = myThread->getTeam()->size();

   /*! initialize the barrier to the current participant number */
   semaphores.resize( numParticipants );

   /*! now we can compute the number of steps of the algorithm */
   q = ( int ) ceil( log2( ( double ) numParticipants ) );

   for ( int s = 0; s < q; s++ ) {
      //compute the current step neighbour id
      int toSign = ( myID + ( int ) pow( 2,s ) ) % numParticipants;

      //wait for the neighbour sem to reach the previous value
      Scheduler::blockOnCondition( &semaphores[toSign].override(), 0 );
      ( semaphores[toSign] )--;

      /*!
         Wait for the semaphore to be signaled for this round
         (check if it reached the number of step signals (+1 because the for starts from 0 for a correct computation of
         neighbours)
       */
      Scheduler::blockOnCondition( &semaphores[myID].override(), -1 );
      ( semaphores[myID] ) + 1;
   }

   /*! at the end of the protocol, we are guaranteed that the semaphores are all 0 */
}




Barrier * createDisseminationBarrier()
{
   return new disseminationBarrier();
}



/*! \class DisseminationBarrierPlugin
    \brief plugin of the related disseminationBarrier class
    \see disseminationBarrier
*/

class DisseminationBarrierPlugin : public Plugin
{

   public:
      DisseminationBarrierPlugin() : Plugin( "Dissemination Barrier Plugin",1 ) {}

      virtual void init() {
         sys.setDefaultBarrFactory( createDisseminationBarrier );
      }
};

DisseminationBarrierPlugin NanosXPlugin;
