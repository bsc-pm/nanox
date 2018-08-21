/*************************************************************************************/
/*      Copyright 2009-2018 Barcelona Supercomputing Center                          */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <https://www.gnu.org/licenses/>.            */
/*************************************************************************************/

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
   ops->addOp();
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
      //if ( sys.getVerboseDevOps()){ 
      //   std::cerr << "memcpy( " << (void*)(_dst + count) << ", " << (void*)(_src + count *_ld) << ", " << _len << " ) [ld= " << _ld << " count= " << _count << " _dst= " << (void*)_dst << " _src= " << (void*)_src << " ]" << std::endl;
      //}
      if (sys._watchAddr != NULL ) {
         if ((uint64_t )sys._watchAddr >= (uint64_t)(_dst + count *_ld ) && (uint64_t )sys._watchAddr < (uint64_t)(_dst + count *_ld + _len)) {
            char buff[256];
            snprintf(buff, 256, "WATCH update: old value %a", *((double *) sys._watchAddr ) );
            *myThread->_file << buff << std::endl;
         }
         if ((uint64_t )sys._watchAddr >= (uint64_t)(_src + count *_ld ) && (uint64_t )sys._watchAddr < (uint64_t)(_dst + count * _ld + _len)) {
            char buff[256];
            snprintf(buff, 256, "WATCH read: value %a", *((double *) sys._watchAddr ) );
            *myThread->_file << buff << std::endl;
         }
      }
      ::memcpy( _dst + count * _ld, _src + count * _ld, _len );
      if (sys._watchAddr != NULL ) {
         if ((uint64_t )sys._watchAddr >= (uint64_t)(_dst + count *_ld ) && (uint64_t )sys._watchAddr < (uint64_t)(_dst + count * _ld + _len)) {
            char buff[256];
            snprintf(buff, 256, "WATCH update: new value %a", *((double *) sys._watchAddr ) );
            *myThread->_file << buff << std::endl;
         }
      }
   }
   //*myThread->_file << "Execueted op " << (void *) _dst  << " ops: " << (void *) _ops << " is in " << _in << " content (dst): [" << ((double *)_dst)[0] << " " << ((double *)_dst)[1] << "]" << std::endl; 
   _ops->completeOp();
   NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( _in ? key_in : key_out, (nanos_event_value_t) 0 ); )
}

#define CHUNK_SIZE 4096

SMPTransferQueue::SMPTransferQueue() : _lock(), _transfers() {}
void SMPTransferQueue::addTransfer( DeviceOps *ops, char *dst, char *src, std::size_t len, std::size_t count, std::size_t ld, bool in ) {
   _lock.acquire();
   // NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   // NANOS_INSTRUMENT ( static nanos_event_key_t key = ID->getEventKey("cache-copy-out"); )
   // NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent( key, (nanos_event_value_t) 4 ); )
   if ( len * count > CHUNK_SIZE*2 && ( true /* FRAGMENT FLAG */ ) ) {
      if ( count == 1 ) {
         std::size_t current_chunk = CHUNK_SIZE;
         std::size_t total_processed = 0;
         while ( total_processed < len ) {
            _transfers.push_back( SMPTransfer(ops, dst+total_processed, src+total_processed, current_chunk, 1, ld, in) );
            total_processed += current_chunk;
            current_chunk = total_processed + (CHUNK_SIZE*2) > len ? CHUNK_SIZE : len - total_processed;
         }
      } else {
         if ( len > CHUNK_SIZE*2 ) {
            for ( std::size_t count_idx = 0; count_idx < count; count_idx += 1 ) {
               std::size_t current_line_chunk = CHUNK_SIZE;
               std::size_t total_line = 0;
               while ( total_line < len ) {
                  _transfers.push_back( SMPTransfer(ops, dst+(ld*count_idx)+total_line, src+(ld*count_idx)+total_line, current_line_chunk, 1, ld, in) );
                  total_line += current_line_chunk;
                  current_line_chunk = total_line + (CHUNK_SIZE*2) > len ?  len - total_line : CHUNK_SIZE;
               }
            }
         } else {
            std::size_t current_count_chunk = CHUNK_SIZE / len;
            std::size_t total_count = 0;
            while ( total_count < count ) {
               _transfers.push_back( SMPTransfer(ops, dst+(ld*total_count), src+(ld*total_count), len, current_count_chunk, ld, in) );
               total_count += current_count_chunk;
               current_count_chunk = total_count + (CHUNK_SIZE/len)*2 > count ? count - total_count : (CHUNK_SIZE/len);
            }
         }
      }
   } else {
      _transfers.push_back( SMPTransfer(ops, dst, src, len, count, ld, in) );
   }
   // NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent( key, (nanos_event_value_t) 0 ); )
   _lock.release();
}
void SMPTransferQueue::tryExecuteOne() {
   if ( !_transfers.empty() ) {
      if ( true /*_lock.tryAcquire()*/ ) {
         _lock.acquire();
         bool found = false;
         SMPTransfer t;
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

} // namespace nanos

#endif
