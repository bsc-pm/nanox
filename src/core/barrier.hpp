#ifndef __NANOS_BARRIER_H
#define __NANOS_BARRIER_H

namespace nanos {

class Barrier {
public:
   Barrier() {};
   virtual void init() = 0;
   virtual void barrier() = 0;
   virtual ~Barrier() {}
};
}

#endif
