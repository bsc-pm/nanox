#ifndef _NANOS_ATOMIC
#define _NANOS_ATOMIC

/* TODO: move to configure
#include <ext/atomicity.h>
#ifndef _GLIBCXX_ATOMIC_BUILTINS
#error "Atomic gcc builtins support is mandatory at this point"
#endif
*/

namespace nanos {

template<typename T>
class Atomic {
private:
    volatile T     value;
public:
    // constructor
    Atomic () {}
    Atomic (T init) : value(init) {}
    // copy constructor
    Atomic (const Atomic &atm) : value(atm.value) {}
    // assignment operator
    const Atomic & operator= (const Atomic &atm);
    const Atomic & operator= (const T val);
    // destructor
    ~Atomic() {}
    
    T operator+ (T val) { return __sync_fetch_and_add(&value,val); }
    T operator++ (int val) { return __sync_fetch_and_add(&value,1); }
    T operator- (T val) { return __sync_fetch_and_sub(&value,val); }
    T operator-- (int val) { return __sync_fetch_and_sub(&value,1); }
    Atomic<T> &operator++ () { __sync_add_and_fetch(&value,1); return *this; }
    Atomic<T> &operator-- () { __sync_sub_and_fetch(&value,1); return *this; }

    operator const volatile T& () const { return value; }

    volatile T & override () { return value; }
};

template<typename T>
const Atomic<T> & Atomic<T>::operator= (const T val)
{
    // TODO: make this an atomic update
    return *this;
}

template<typename T>
const Atomic<T> & Atomic<T>::operator= (const Atomic<T> &val)
{
    return operator=(val.value);
}

class Lock {
private:
    typedef enum { FREE=0, BUSY=1 } state_t;
    
    volatile state_t      state;

    // disable copy constructor and assignment operator
    Lock(const Lock &lock);
    const Lock & operator= (const Lock&);
public:
    // constructor
    Lock(state_t init=FREE) : state(init) {};
    // destructor
    ~Lock() {}

    void acquire (void);
    bool tryAcquire (void);
    void release (void);
    
    const state_t operator* () const { return state; }
    const state_t getState () const { return state; }

    void operator++ (int val) { acquire(); }
    void operator-- (int val) { release(); }
};

inline void Lock::acquire (void)
{
spin:
        while ( state == BUSY );
        if (__sync_lock_test_and_set(&state,BUSY)) goto spin;
}

inline bool Lock::tryAcquire (void)
{
      if ( state == FREE ) {
	  if (__sync_lock_test_and_set(&state,BUSY)) return false;
	  else return true;
      } else return false;
}

inline void Lock::release (void)
{
    __sync_lock_release(&state);
}

inline void memory_fence () { __sync_synchronize(); }

inline bool compare_and_swap(int *ptr, int oldval, int newval) { return __sync_bool_compare_and_swap (ptr, oldval, newval);}

};

#endif
