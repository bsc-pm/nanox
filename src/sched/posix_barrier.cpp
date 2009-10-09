#include <assert.h>
#include <pthread.h>

#include "barrier.hpp"
#include "system.hpp"
#include "atomic.hpp"


using namespace nanos;

class posixBarrier: public Barrier
{
private:
   pthread_barrier_t pBarrier;

public:
   posixBarrier();
   posixBarrier(int groupSize);
   void init();
   void barrier();
   void setSize(int size);
};


//For now the size is put at the system creation, because it is not guaranteed that the basethread is created when we instatiate the barrier
//centralizedBarrier::centralizedBarrier() {
 //  sem = new Atomic<int>( sys.getSGSize() );
//}


posixBarrier::posixBarrier(int groupSize) {
   assert(groupSize > 0);
   unsigned numThreads = (unsigned) groupSize;
   pthread_barrier_init ( &pBarrier, NULL, numThreads );
}

void posixBarrier::init() {}

void posixBarrier::setSize(int size) { 
   assert(size > 0);
   unsigned numThreads = (unsigned) size;
   assert(pthread_barrier_init ( &pBarrier, NULL, numThreads ) != 0);
}

void posixBarrier::barrier() {
   pthread_barrier_wait(&pBarrier);
}


//factories
//Same of above for void constructor
// Barrier * createCentralizedBarrier() {
//    return new centralizedBarrier();
// }

Barrier * createPosixBarrier(int gSize) {
   return new posixBarrier(gSize);
}

