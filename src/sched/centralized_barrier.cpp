#include "barrier.hpp"
#include "system.hpp"
#include "atomic.hpp"

using namespace nanos;

class centralizedBarrier: public Barrier
{
private:
   Atomic<int> * sem;

public:
   centralizedBarrier();
   centralizedBarrier(int groupSize);
   void init();
   void barrier();
   void setSize(int size);
};


//For now the size is put at the system creation, because it is not guaranteed that the basethread is created when we instatiate the barrier
//centralizedBarrier::centralizedBarrier() {
 //  sem = new Atomic<int>( sys.getSGSize() );
//}


centralizedBarrier::centralizedBarrier(int groupSize) {
   sem = new Atomic<int>( groupSize );
}

void centralizedBarrier::init() {}

void centralizedBarrier::setSize(int size) { (*sem) = size; }

void centralizedBarrier::barrier() {
   if((*sem)-- != 0)
      while((*sem) != 0) {}
   //the last one decrementing the semaphore also resets it to the current groupSize (for dynamic support)
   else (*sem) = sys.getSGSize(); 
}


//factories
//Same of above for void constructor
// Barrier * createCentralizedBarrier() {
//    return new centralizedBarrier();
// }

Barrier * createCentralizedBarrier(int gSize) {
   return new centralizedBarrier(gSize);
}

