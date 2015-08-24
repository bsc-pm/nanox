#ifndef SMPTRANSFERQUEUE
#define SMPTRANSFERQUEUE
#include "smptransferqueue_decl.hpp"
#include "atomic.hpp"
#include "deviceops.hpp"
namespace nanos {

SMPTransfer::SMPTransfer() :
   _ops((DeviceOps *) NULL),
   _dst((char *)0xdeadbeef),
   _src((char *)0xbeefdead),
   _len(133),
   _count(0),
   _ld(0),
   _in( false ) {
}

SMPTransfer::SMPTransfer( DeviceOps *ops, char *dst, char *src, std::size_t len, std::size_t count, std::size_t ld, bool in ) : _ops(ops), _dst(dst), _src(src), _len(len), _count(count), _ld(ld), _in( in ) {
}
SMPTransfer::SMPTransfer( SMPTransfer const &s ) : _ops(s._ops), _dst(s._dst), _src(s._src), _len(s._len), _count(s._count), _ld(s._ld), _in(s._in) {
}
SMPTransfer &SMPTransfer::operator=( SMPTransfer const &s ) {
   _ops = s._ops;
   _dst = s._dst;
   _src = s._src;
   _len = s._len;
   _count = s._count;
   _ld = s._ld;
   _in = s._in;
   return *this;
}
SMPTransfer::~SMPTransfer() {}

void SMPTransfer::execute() {
   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t key_in = ID->getEventKey("cache-copy-in"); )
   NANOS_INSTRUMENT ( static nanos_event_key_t key_out = ID->getEventKey("cache-copy-out"); )
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( _in ? key_in : key_out , (nanos_event_value_t) _count * _len ); )
   for ( std::size_t count = 0; count < _count; count += 1) {
      ::memcpy( _dst + count * _ld, _src + count * _ld, _len );
   }
   _ops->completeOp();
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( _in ? key_in : key_out, (nanos_event_value_t) 0 ); )
}


SMPTransferQueue::SMPTransferQueue() : _lock(), _transfers() {}
void SMPTransferQueue::addTransfer( DeviceOps *ops, char *dst, char *src, std::size_t len, std::size_t count, std::size_t ld, bool in ) {
   _lock.acquire();
   // NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   // NANOS_INSTRUMENT ( static nanos_event_key_t key = ID->getEventKey("cache-copy-out"); )
   // NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( key, (nanos_event_value_t) 4 ); )
   _transfers.push_back( SMPTransfer(ops, dst, src, len, count, ld, in) );
   // NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( key, (nanos_event_value_t) 0 ); )
   _lock.release();
}
void SMPTransferQueue::tryExecuteOne() {
   if ( !_transfers.empty() ) {
      if ( true /*_lock.tryAcquire()*/ ) {
         _lock.acquire();
         bool found = false;
         SMPTransfer t( (DeviceOps *) NULL, (char *) 0xdeadbeef, (char *) 0xbeefdead, (std::size_t) 133, (std::size_t) 0, (std::size_t) 0, false);
         if ( !_transfers.empty() ) {
            found = true;
            t = _transfers.front();
            _transfers.pop_front();
         }
         _lock.release();
         if ( found ) 
            t.execute();
      }
   }
}

}
#endif
