#include <assert.h>
#include <vector>
#include <math.h>

#include "barrier.hpp"
#include "system.hpp"
#include "atomic.hpp"


using namespace nanos;
using namespace std;

class disseminationBarrier: public Barrier
{
private:
   vector<Atomic<int> > semaphores;
   int threadNum, q; //q is the log of threadNum
public:
   disseminationBarrier();
   disseminationBarrier(int groupSize);
   void init();
   void barrier();
   void setSize(int size);
};


//TODO: number of threads must be a power of 2: remove this hypothesis
disseminationBarrier::disseminationBarrier(int groupSize) {
   assert(groupSize > 0);
   threadNum = groupSize;
   q = (int) log(threadNum);
   semaphores.resize(groupSize);
   vector<Atomic<int> >::iterator it;
   //also initializes to 1 the semaphore
   for(it = semaphores.begin(); it != semaphores.end(); it++) {
      (*it) = 1;
   }
}

void disseminationBarrier::init() {}

void disseminationBarrier::setSize(int size) { 
   assert(size > 0);
   threadNum = size;
   q = (int) log(threadNum);
   semaphores.resize(size);
   vector<Atomic<int> >::iterator it;
   //also initializes to 1 the semaphore
   for(it = semaphores.begin(); it != semaphores.end(); it++) {
      (*it) = 1;
   }
}

void disseminationBarrier::barrier() {
   //for now, suppose that we can obtain an ID
   int myID = -1;
   for(int s = 0; s < q; s++) {
      int toSign = myID + (int) pow(2,s);
      (semaphores[toSign])--;
      while(semaphores[myID] != 0) {}
      if(s < q) {
         //reset the semaphore for the next round
         semaphores[myID]++; //notice that we allow another thread to signal it before we reset it (not set to 1!)
      }
   }
   //reset the semaphore for the next barrier
   semaphores[myID] = 1;
}


//factories
//Same of above for void constructor
// Barrier * createCentralizedBarrier() {
//    return new centralizedBarrier();
// }

Barrier * createDisseminationBarrier(int gSize) {
   return new disseminationBarrier(gSize);
}

