#ifndef _NANOS_THREAD_TEAM
#define _NANOS_THREAD_TEAM

#include <vector>
#include "basethread.hpp"
#include "schedule.hpp"
#include "barrier.hpp"


namespace nanos {

class ThreadTeam {
   private:
      std::vector<BaseThread *> threads;
      int  idleThreads;
      int  numTasks;
      Barrier * barrAlgorithm;

      // disable copy constructor & assignment operation
      ThreadTeam(const ThreadTeam &sys);
      const ThreadTeam & operator= (const ThreadTeam &sys);

   public:

      ThreadTeam ( int maxThreads, SG &policy ) : idleThreads(0), numTasks(0) { threads.reserve(maxThreads); }
      
      unsigned size() const { return threads.size(); }

      const BaseThread & operator[]  ( int i ) const { return *threads[i]; }
      BaseThread & operator[]  ( int i ) { return *threads[i]; }

      void addThread (BaseThread *thread) {
          threads.push_back(thread);
      }

      void setBarrAlgorithm(Barrier * barrAlg) { barrAlgorithm = barrAlg; }
};

}

#endif
