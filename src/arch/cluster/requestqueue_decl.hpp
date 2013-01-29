#ifndef REQUESTQUEUE_DECL_H
#define REQUESTQUEUE_DECL_H
#include <list>
#include <map>
#include "atomic_decl.hpp"
namespace nanos {
template <class T>
class RequestQueue {
   std::list< T * > _queue;
   Lock _lock;
   public:
   RequestQueue();
   ~RequestQueue();
   void add( T * elem );
   T *fetch();
   T *tryFetch();
};

template <class T>
class RequestMap {
   std::map< uint64_t, T * > _map;
   Lock _lock;
   public:
   RequestMap();
   ~RequestMap();
   void add( uint64_t key, T * elem );
   T *fetch( uint64_t key );
   T *tryFetch( uint64_t key );
};
}
#endif /* REQUESTQUEUE_DECL_H */
