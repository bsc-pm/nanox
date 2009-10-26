#include <assert.h>
#include <vector>
#include <math.h>

#include "barrier.hpp"
#include "system.hpp"
#include "atomic.hpp"


using namespace nanos;
using namespace std;


/*! 
   \class disseminationBarrier 
   \brief implements a barrier according to the dissemination algorithm

 */
class disseminationBarrier: public Barrier
{
private:
   vector<Atomic<int> > semaphores;
   int q; //q is the log of threadNum
public:
   disseminationBarrier(int numP);
   void init();
   void barrier();
   void barrier(int id);
};


disseminationBarrier::disseminationBarrier(int numP): Barrier(numP) {
   assert(numP > 0);
   q = (int) ceil(log2(numP));

   /*! semaphores are automatically initialized to their default value (\see Atomic)*/
   semaphores.resize(numP);
}

void disseminationBarrier::init() {}

void disseminationBarrier::barrier() {
   int myID = -1;
   for(int s = 0; s < q; s++) {
      int toSign = ( myID + (int) pow(2,s) ) % numParticipants;
      (semaphores[toSign])--;
      while(semaphores[myID] != 0) {}
      if(s < q) {
         //reset the semaphore for the next round
         semaphores[myID]++; //notice that we allow another thread to signal it before we reset it (not set to 1!)
      }
   }
   //reset the semaphore for the next barrier
   //warning: as we do not have an atomic assignement, we use the substraction operation
   semaphores[myID]-1;
}


void disseminationBarrier::barrier(int myID) {
   for(int s = 0; s < q; s++) {
      //compute the current step neighbour id
      int toSign = ( myID + (int) pow(2,s) ) % numParticipants;

      //wait for the neighbour sem to reach the previous value
      Scheduler::blockOnCondition( &semaphores[toSign].override(), 0 );
      (semaphores[toSign])--;

      /*! 
         Wait for the semaphore to be signaled for this round 
         (check if it reached the number of step signals (+1 because the for starts from 0 for a correct computation of neighbours) 
       */
      Scheduler::blockOnCondition( &semaphores[myID].override(), -1 );
      (semaphores[myID]) + 1;
   }
   /*! at the end of the protocol, we are guaranteed that the semaphores are all 0 */
}




Barrier * createDisseminationBarrier(int numP) {
   return new disseminationBarrier(numP);
}
