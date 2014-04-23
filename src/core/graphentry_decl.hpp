#ifndef GRAPHENTRY_H
#define GRAPHENTRY_H
#include <iostream>

namespace nanos
{
   class GraphEntry {
      unsigned int _id;
      unsigned int _peid;
      unsigned int _count;
      unsigned int _node;
      bool _isWait;
      public:
         GraphEntry ( int id ) : _id ( id ), _peid( 0 ), _count(0), _node(0), _isWait( false ) {}
         unsigned int getId() { return _id; }
         void setPeId(unsigned int pe) { _peid = pe; }
         unsigned int getPeId() { return _peid; }
         void setIsWait() { _isWait = true; }
         void setNoWait() { _isWait = false; }
         bool isWait() { return _isWait; }
         unsigned int getCount() { return _count; }
         void setCount(unsigned int c) { _count = c; }
         void setNode(unsigned int n) { _node = n; }
         unsigned int getNode() { return _node; }
         friend std::ostream & operator<<(std::ostream &io, GraphEntry const &ge);
   };


   inline std::ostream & operator<<(std::ostream &io, GraphEntry const &ge) {
      if (!ge._isWait)
      {
         io << ge._id ;
      }
      else
      {
         io << "Wait" << ge._count;
      }
      return io;
   }
}

#endif
