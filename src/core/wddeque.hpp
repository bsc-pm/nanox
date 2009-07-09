#ifndef _NANOS_LIB_WDDEQUE
#define _NANOS_LIB_WDDEQUE

#include <list>
#include "atomic.hpp"
#include "debug.hpp"
#include "workdescriptor.hpp"
#include "basethread.hpp"

namespace nanos {

class SchedulePredicate {
    bool operator() (WorkDescriptor *wd);
};

class WDDeque {
private:
   typedef std::list<WorkDescriptor *> deque_t;

   deque_t	dq;
   Lock 	lock;

   //TODO: copy constructor, assignment
   WDDeque (const WDDeque &);
   const WDDeque & operator= (const WDDeque &);
   
public:
    WDDeque() {}
    ~WDDeque() { /*TODO: ensure queue is empty*/ }

    void push_front (WorkDescriptor *wd);
    void push_back(WorkDescriptor *wd);
    WorkDescriptor * pop_front (BaseThread *thread);
    //TODO: WorkDescriptor * pop_front (PE *pe, SchedulePredicate &predicate);
    //TODO: pop_back
    WorkDescriptor * pop_back (BaseThread *thread);

    bool removeWD(WorkDescriptor * toRem);
};

inline void WDDeque::push_front (WorkDescriptor *wd)
{
  wd->setMyQueue(this);
  lock++;
  dq.push_back(wd);
  memory_fence();
  lock--;
}

inline void WDDeque::push_back (WorkDescriptor *wd)
{
  wd->setMyQueue(this);
  lock++;
  dq.push_back(wd);
  memory_fence();
  lock--;
}

// Only ensures tie semantics
inline WorkDescriptor * WDDeque::pop_front (BaseThread *thread)
{
    WorkDescriptor *found = NULL;

    if ( dq.empty() ) return NULL;
    memory_fence();
    lock++;
    if (!dq.empty()) {
      WDDeque::deque_t::iterator it;

      for ( it = dq.begin() ; it != dq.end(); it++ )
      {
           if ( !(*it)->isTied() || (*it)->isTiedTo() == thread ) {
	        found = *it;
		dq.erase(it);
	        break;
	   }
      }
    }
    lock--;

    ensure(!found || !found->isTied() || found->isTiedTo() == thread, "" );

    if(found != NULL) {found->setMyQueue(NULL);}

    return found;
}


// Only ensures tie semantics
inline WorkDescriptor * WDDeque::pop_back (BaseThread *thread)
{
    WorkDescriptor *found = NULL;

    if ( dq.empty() ) return NULL;
    memory_fence();
    lock++;
    if (!dq.empty()) {
      WDDeque::deque_t::iterator it;

	for(it = dq.end(), --it; it != dq.begin(); --it)
	{
           if ( !(*it)->isTied() || (*it)->isTiedTo() == thread ) {
	        found = *it;
		dq.erase(it);
	        break;
	   }
	}
    }
    lock--;

    ensure(!found || !found->isTied() || found->isTiedTo() == thread, "" );

    if(found != NULL) {found->setMyQueue(NULL);}
    return found;
}




inline bool WDDeque::removeWD(WorkDescriptor * toRem)
{
	if ( dq.empty() ) return false;
	memory_fence();
	lock++;
	
	if (!dq.empty() && toRem->getMyQueue() == this) {
		WDDeque::deque_t::iterator it;			

		for(it = dq.begin(); it != dq.end(); it++) {
			if(*it == toRem) {
				dq.erase(it);
				toRem->setMyQueue(NULL);

				lock--;
				return true;
			}
		}
	}
	lock--;

	return false;	
}


}


#endif

