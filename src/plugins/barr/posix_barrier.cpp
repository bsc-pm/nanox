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
   posixBarrier(int numP);
   void init();
   void barrier();
};


posixBarrier::posixBarrier(int numP): Barrier(numP) {
   assert(numP > 0);
   unsigned numThreads = (unsigned) numP;
   pthread_barrier_init ( &pBarrier, NULL, numThreads );
}

void posixBarrier::init() {}

void posixBarrier::barrier() {
   pthread_barrier_wait(&pBarrier);
}


Barrier * createPosixBarrier(int numP) {
   return new posixBarrier(numP);
}

