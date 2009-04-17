#ifndef _NANOS_LIB_QUEUE
#define _NANOS_LIB_QUEUE

#include <queue>
#include "atomic.hpp"
#include "debug.hpp"

namespace nanos {

// FIX: implement own queue without coherence problems? lock-free?
template<typename T> class Queue {
private:
	Lock qLock;
	std::queue<T>  q;

      // disable copy constructor and assignment operator
      Queue(Queue &orig);
      const Queue & operator= (const Queue &orig);
public:
      // constructors
      Queue() {}
      // destructor
      ~Queue() {}

      void push(T data);
      T    pop (void);
      bool try_pop (T& result);
};

template<typename T> void Queue<T>::push (T data)
{
  qLock++;
  q.push(data);
  memory_fence();
  qLock--;
}

template<typename T> T Queue<T>::pop (void)
{
spin:
    while (q.empty()) memory_fence();
  // not empty
    qLock++;
    if (!q.empty()) {
      T tmp = q.front();
      q.pop();
      qLock--;
      return tmp;
    }
    qLock--;
    goto spin;
}

template<typename T> bool Queue<T>::try_pop (T& result)
{
    bool found = false;
   
    if (q.empty()) return false;
    memory_fence();
    qLock++;
    if (!q.empty()) {
      result = q.front();
      q.pop();
      found = true;
    }
    qLock--;

    return found;
}

};

#endif

