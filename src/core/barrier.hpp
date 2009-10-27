#ifndef __NANOS_BARRIER_H
#define __NANOS_BARRIER_H

namespace nanos {

class Barrier {
protected:
   int numParticipants;
public:
   Barrier(int numP): numParticipants(numP) {};
   virtual void init() = 0;
   virtual void barrier() = 0;
   virtual ~Barrier() {}
};


typedef Barrier * (*barrFactory) (int numParticipants);


}

#endif
