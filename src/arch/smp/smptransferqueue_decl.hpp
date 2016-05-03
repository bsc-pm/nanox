#ifndef SMPTRANSFERQUEUE_DECL
#define SMPTRANSFERQUEUE_DECL

#include <list>
#include "atomic_decl.hpp"
#include "deviceops_fwd.hpp"

namespace nanos {

class SMPTransfer {
   DeviceOps   *_ops;
   char        *_dst;
   char        *_src;
   std::size_t  _len;
   std::size_t  _count;
   std::size_t  _ld;
   bool         _in;
   public:
   SMPTransfer();
   SMPTransfer( DeviceOps *ops, char *dst, char *src, std::size_t len, std::size_t count, std::size_t ld, bool in );
   SMPTransfer( SMPTransfer const &s );
   SMPTransfer &operator=( SMPTransfer const &s );
   ~SMPTransfer();
   void execute();
};

class SMPTransferQueue {
   Lock _lock;
   std::list< SMPTransfer > _transfers;
   public:
   SMPTransferQueue();
   void addTransfer( DeviceOps *ops, char *dst, char *src, std::size_t len, std::size_t count, std::size_t ld, bool in );
   void tryExecuteOne();
};

} // namespace nanos

#endif
