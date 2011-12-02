#ifndef MEMORYMAP_H
#define MEMORYMAP_H

#include "memorymap_decl.hpp"

namespace nanos {

inline MemoryChunk& MemoryChunk::operator=( const MemoryChunk &mc ) {
   _addr = mc._addr;
   _len = mc._len;
   return *this;
}

inline bool MemoryChunk::operator<( const MemoryChunk &chunk ) const {
   return _addr < chunk._addr;
} 

inline uint64_t MemoryChunk::getAddress() const {
   return _addr;
}
inline std::size_t MemoryChunk::getLength() const {
   return _len;
}

inline bool MemoryChunk::equal( const MemoryChunk &target ) const {
   return ( _addr == target._addr && _len == target._len );
}

inline bool MemoryChunk::contains( const MemoryChunk &target ) const {
   return ( ( _addr <= target._addr) && ( ( _addr + _len ) >= ( target._addr + target._len ) ) );
}

inline MemoryChunk::OverlapType MemoryChunk::checkOverlap( const MemoryChunk &target ) const {
   OverlapType ret;
   if ( _addr < target._addr ) {
      if ( _addr + _len > target._addr ) {
         if ( _addr + _len > target._addr + target._len ) {
            ret = SUBCHUNK_OVERLAP; 
         } else if ( _addr + _len < target._addr + target._len ) {
            ret = END_OVERLAP;
         } else {
            ret = SUBCHUNK_END_OVERLAP; 
         }
      } else {
         ret = NO_OVERLAP;
      }
   } else if ( _addr > target._addr ) {
      if ( target._addr + target._len > _addr ) {
         if ( target._addr + target._len > _addr + _len ) {
            ret = TOTAL_OVERLAP; 
         } else if ( target._addr + target._len < _addr + _len ) {
            ret = BEGIN_OVERLAP;
         } else {
            ret = TOTAL_END_OVERLAP;
         }
      } else {
         ret = NO_OVERLAP;
      }
   } else { //addr == target._addr
      if ( target._len > _len ) {
         ret = TOTAL_BEGIN_OVERLAP;
      } else {
         ret = SUBCHUNK_BEGIN_OVERLAP;
      }
   }
   return ret;
}

inline void MemoryChunk::intersect( MemoryChunk &mcA, MemoryChunk &mcB, MemoryChunk &mcC ) {
   //assume A < B and NO TOTAL overlap
   std::size_t intersectionLen = mcA._addr + mcA._len - mcB._addr;
   mcC._addr = mcA._addr + mcA._len;
   mcC._len = mcB._len - intersectionLen;
   mcB._len = intersectionLen;
   mcA._len -= intersectionLen;
}

inline void MemoryChunk::partition( MemoryChunk &mcA, MemoryChunk &mcB, MemoryChunk &mcC ) {
   //assume A < B and A totally overlaps B
   std::size_t remainingLen = ( mcA._addr + mcA._len ) - ( mcB._addr + mcB._len );
   mcC._addr = mcB._addr + mcB._len;
   mcC._len = remainingLen;
   mcA._len -= ( remainingLen + mcB._len );
}

inline void MemoryChunk::partitionBeginAgtB( MemoryChunk &mcA, MemoryChunk &mcB ) {
   //assume A.addr = B.addr, A.len > B.len,  A.addr is NOT modified 
   std::size_t bLen = mcA._len - mcB._len;
   mcB._addr = mcA._addr + mcB._len;
   mcB._len = bLen;
   mcA._len -= mcB._len;
}

inline void MemoryChunk::partitionBeginAltB( MemoryChunk &mcA, MemoryChunk &mcB ) {
   //assume A.addr = B.addr, A.len < B.len,  A.addr is NOT modified 
   std::size_t bLen = mcB._len - mcA._len;
   mcB._addr = mcA._addr + mcA._len;
   mcB._len = bLen;
}

inline void MemoryChunk::partitionEnd( MemoryChunk &mcA, MemoryChunk &mcB ) {
   //assume A.addr+A.len = B.addr+B.len, A.len > B.len,  B.addr is NOT modified 
   mcA._len -= mcB._len;
}

inline void MemoryChunk::expandIncluding( const MemoryChunk &mcB ) {
   _len = ( mcB._addr + mcB._len ) - _addr;
}

inline void MemoryChunk::expandExcluding( const MemoryChunk &mcB ) {
   _len = ( mcB._addr - _addr );
}

inline void MemoryChunk::cutAfter( const MemoryChunk &mc ) {
   _len = ( _addr + _len ) - ( mc._addr + mc._len );
   _addr = mc._addr + mc._len;
}

template < typename _Type >
void MemoryMap< _Type >::processNoOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   _Type *ptr = (_Type *) NULL;
   //message0("NO_OVERLAP detected " << (void *) iterKey.getAddress() << " : " << iterKey.getLength() << " versus " << (void *) hint->first.getAddress() << " : " << hint->first.getLength() );
   hint = this->insert( hint, typename BaseMap::value_type( key, ptr ) );
   ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) );
}

template < typename _Type >
void MemoryMap< _Type >::processBeginOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   /*
    *                +=====================+
    *                |     rightChunk      |
    *                | (already inserted)  |
    *                +=====================+
    *   +-----------------------+
    *   |      leftChunk        |
    *   |    (to be inserted)   |
    *   +-----------------------+
    *           == intersect ==
    *   +------------+
    *   | leftChunk  |
    *   +------------+
    *        ^       +==========+............+
    *        |       |rightChunk|righLeftOver|
    *        |       +==========+............+
    *   added to result----^
    */
   _Type *ptr = (_Type *) NULL;
   MemoryChunk &leftChunk = key;
   MemoryChunk &rightChunk = const_cast< MemoryChunk & >( hint->first );
   MemoryChunk rightLeftOver;
   //message0("BEGIN_OVERLAP detected " << (void *) iterKey.getAddress() << " : " << iterKey.getLength() << " versus " << (void *) hint->first.getAddress() << " : " << hint->first.getLength() );

   MemoryChunk::intersect( leftChunk, rightChunk, rightLeftOver );

   hint = this->insert( hint, typename BaseMap::value_type( leftChunk, ptr ) ); //add the left chunk, 
   ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) );
   hint++; //advance the iterator (it should reach "rightChunk"!)
   ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) );

   hint = this->insert( hint, typename BaseMap::value_type( rightLeftOver, NEW _Type( *hint->second ) ) );
}

template < typename _Type >
void MemoryMap< _Type >::processEndOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   MemoryChunk &leftChunk = const_cast< MemoryChunk & >( hint->first );
   MemoryChunk &rightChunk = key;
   MemoryChunk rightLeftOver;
   /*
    *   +=======================+
    *   |      leftChunk        |
    *   |   (already inserted)  |
    *   +=======================+
    *                +---------------------+
    *                |     rightChunk      |
    *                |  (to be inserted)   |
    *                +---------------------+
    *           == intersect ==
    *   +============+
    *   | leftChunk  |
    *   +============+
    *                +----------+............+
    *                |rightChunk|righLeftOver| <- not to be inserted yet.
    *                +----------+............+
    *   added to result-----^        ^------ not added to result (processed on next iterations)
    */

   MemoryChunk::intersect( leftChunk, rightChunk, rightLeftOver );

   //add the right chunk 
   hint = this->insert( hint, typename BaseMap::value_type( rightChunk, NEW _Type( *hint->second ) ) );
   //result: right chunk
   ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) );
   //leftover not added to result since more overlapping may exist
   key = rightLeftOver;  //try to add what is left
}

template < typename _Type >
void MemoryMap< _Type >::processTotalOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   _Type *ptr = (_Type *) NULL;
   //message0("TOTAL_OVERLAP detected " << (void *) iterKey.getAddress() << " : " << iterKey.getLength() << " versus " << (void *) hint->first.getAddress() << " : " << hint->first.getLength() );
   MemoryChunk &leftChunk = key;
   MemoryChunk &rightChunk = const_cast< MemoryChunk & >( hint->first );
   _Type ** rightChunkDataPtr = &( hint->second );
   MemoryChunk leftLeftOver;
   /*
    *                +=====================+
    *                |     rightChunk      |
    *                |  (already inserted) |
    *                +=====================+
    *   +-----------------------------------------------+
    *   |                  leftChunk                    |
    *   |               (to be inserted)                |
    *   +-----------------------------------------------+
    *                    == partition ==
    *                +=====================+
    *                |     rightChunk      |
    *                +=====================+
    *   +------------+        ^            +............+
    *   | leftChunk  |        |            |leftLeftOver|
    *   +------------+        |            +............+
    *         ^------  added to result          ^-- not added (keep processing)
    */

   MemoryChunk::partition( leftChunk, rightChunk, leftLeftOver );

   hint = this->insert( hint, typename BaseMap::value_type( leftChunk, ptr ) ); //add the left chunk, 
   ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) ); //return: leftChunk
   ptrList.push_back( MemChunkPair ( &rightChunk, rightChunkDataPtr ) ); //return: rightChunk
   key = leftLeftOver; //try to add what is left
}

template < typename _Type >
void MemoryMap< _Type >::processSubchunkOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   MemoryChunk &leftChunk = const_cast< MemoryChunk & >( hint->first );
   MemoryChunk &rightChunk = key;
   MemoryChunk leftLeftOver;
   /*
    *   +===============================================+
    *   |                  leftChunk                    |
    *   |               (already inserted)              |
    *   +===============================================+
    *                +---------------------+
    *                |     rightChunk      |
    *                |  (to be inserted)   |
    *                +---------------------+
    *                    == partition ==
    *   +============+                     +............+
    *   | leftChunk  |                     |leftLeftOver|
    *   +============+                     +............+
    *                +---------------------+
    *                |     rightChunk      |
    *                +---------------------+
    *                   added to result
    */

   MemoryChunk::partition( leftChunk, rightChunk, leftLeftOver );

   hint = this->insert( hint, typename BaseMap::value_type( rightChunk, NEW _Type( *hint->second ) ) ); //add the right chunk, 
   ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) ); //return: rightChunk
   hint = this->insert( hint, typename BaseMap::value_type( leftLeftOver, NEW _Type( *hint->second ) ) ); //add the leftover
}

template < typename _Type >
void MemoryMap< _Type >::processTotalBeginOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   MemoryChunk &leftChunk = const_cast< MemoryChunk & >( hint->first );
   MemoryChunk &rightChunk = key;
   /*
    *   +=====================+
    *   |      leftChunk      |
    *   |  (already inserted) |
    *   +=====================+
    *   +-----------------------------------------------+
    *   |                 rightChunk                    |
    *   |               (to be inserted)                |
    *   +-----------------------------------------------+
    *                    == partitionBeginAltB ==
    *   +=====================+
    *   |      leftChunk      |
    *   +=====================+
    *             ^           +.........................+
    *             |           |       rightChunk        |
    *             |           +.........................+
    *       added to result         ^-- not added (keep processing)
    */
   MemoryChunk::partitionBeginAltB( leftChunk, rightChunk );
   ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) ); //return: leftChunk
   key = rightChunk; // redundant
}

template < typename _Type >
void MemoryMap< _Type >::processSubchunkBeginOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   MemoryChunk &leftChunk = const_cast< MemoryChunk & >( hint->first );
   MemoryChunk &rightChunk = key;
   /*
    *   +===============================================+
    *   |                  leftChunk                    |
    *   |               (already inserted)              |
    *   +===============================================+
    *   +---------------------+
    *   |     rightChunk      |
    *   |  (to be inserted)   |
    *   +---------------------+
    *                == partitionBeginAgtB ==
    *                         +-------------------------+
    *                         |        rightChunk       |
    *                         +-------------------------+
    *   +=====================+
    *   |     leftchunk       |
    *   +=====================+
    *       added to result
    */
   MemoryChunk::partitionBeginAgtB( leftChunk, rightChunk );
   ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) ); //return: leftChunk
   hint = this->insert( hint, typename BaseMap::value_type( rightChunk, NEW _Type( *hint->second ) ) ); //add the rightChunk
}

template < typename _Type >
void MemoryMap< _Type >::processTotalEndOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   _Type *ptr = (_Type *) NULL;
   MemoryChunk &leftChunk = key;
   MemoryChunk &rightChunk = const_cast< MemoryChunk & >( hint->first );
   const MemoryChunk *rightChunkPtr = &( hint->first );
   _Type ** rightChunkDataPtr = &( hint->second );
   /*
    *                             +=====================+
    *                             |     rightChunk      |
    *                             |  (already inserted) |
    *                             +=====================+
    *   +-----------------------------------------------+
    *   |                  leftChunk                    |
    *   |               (to be inserted)                |
    *   +-----------------------------------------------+
    *                    == partition ==
    *                             +=====================+
    *                             |     rightChunk      |
    *                             +=====================+
    *   +-------------------------+        ^
    *   |        leftChunk        |        |
    *   +-------------------------+        |
    *         ^------  added to result ----+
    */
   MemoryChunk::partitionEnd( leftChunk, rightChunk );
   hint = this->insert( hint, typename BaseMap::value_type( rightChunk, ptr ) ); //add the leftChunk
   ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) ); //return: leftChunk
   ptrList.push_back( MemChunkPair ( rightChunkPtr, rightChunkDataPtr ) ); //return: rightChunk
}

template < typename _Type >
void MemoryMap< _Type >::processSubchunkEndOverlap( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   MemoryChunk &leftChunk = const_cast< MemoryChunk & >( hint->first );
   MemoryChunk &rightChunk = key;
   /*
    *   +===============================================+
    *   |                  leftChunk                    |
    *   |               (already inserted)              |
    *   +===============================================+
    *                             +---------------------+
    *                             |     rightChunk      |
    *                             |  (to be inserted)   |
    *                             +---------------------+
    *                    == partition ==
    *   +=========================+
    *   |        leftChunk        |
    *   +=========================+
    *                             +---------------------+
    *                             |     rightChunk      |
    *                             +---------------------+
    *                                added to result
    */
   MemoryChunk::partitionEnd( leftChunk, rightChunk );
   hint = this->insert( hint, typename BaseMap::value_type( rightChunk, NEW _Type( *hint->second ) ) ); //add the rightChunk
   ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) ); //return: rightChunk
}

template < typename _Type >
void MemoryMap< _Type >::processNoOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   //_Type *ptr = (_Type *) NULL;
   //message0("NO_OVERLAP detected " << (void *) iterKey.getAddress() << " : " << iterKey.getLength() << " versus " << (void *) hint->first.getAddress() << " : " << hint->first.getLength() );
   //hint = this->insert( hint, typename BaseMap::value_type( key, ptr ) );
   ptrList.push_back( MemChunkPair ( NEW MemoryChunk( key ), NULL ) );
}

template < typename _Type >
void MemoryMap< _Type >::processBeginOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   /*
    *                +=====================+
    *                |     rightChunk      |
    *                | (already inserted)  |
    *                +=====================+
    *   +-----------------------+
    *   |      leftChunk        |
    *   |    (to be inserted)   |
    *   +-----------------------+
    *           == intersect ==
    *   +------------+
    *   | leftChunk  |
    *   +------------+
    *        ^       +==========+............+
    *        |       |rightChunk|righLeftOver|
    *        |       +==========+............+
    *   added to result----^
    */
   //_Type *ptr = (_Type *) NULL;
   MemoryChunk leftChunk = key;
   MemoryChunk rightChunk = hint->first ;
   MemoryChunk rightLeftOver;
   //message0("BEGIN_OVERLAP detected " << (void *) iterKey.getAddress() << " : " << iterKey.getLength() << " versus " << (void *) hint->first.getAddress() << " : " << hint->first.getLength() );

   MemoryChunk::intersect( leftChunk, rightChunk, rightLeftOver );

   //hint = this->insert( hint, typename BaseMap::value_type( leftChunk, ptr ) ); //add the left chunk, 
   ptrList.push_back( MemChunkPair ( NEW MemoryChunk( leftChunk ) , NULL ) );
   //hint++; //advance the iterator (it should reach "rightChunk"!)
   ptrList.push_back( MemChunkPair ( NEW MemoryChunk( rightChunk ) , &(hint->second) ) );

   //hint = this->insert( hint, typename BaseMap::value_type( rightLeftOver, NEW _Type( *hint->second ) ) );
}

template < typename _Type >
void MemoryMap< _Type >::processEndOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   MemoryChunk leftChunk = hint->first ;
   MemoryChunk rightChunk = key;
   MemoryChunk rightLeftOver;
   /*
    *   +=======================+
    *   |      leftChunk        |
    *   |   (already inserted)  |
    *   +=======================+
    *                +---------------------+
    *                |     rightChunk      |
    *                |  (to be inserted)   |
    *                +---------------------+
    *           == intersect ==
    *   +============+
    *   | leftChunk  |
    *   +============+
    *                +----------+............+
    *                |rightChunk|righLeftOver| <- not to be inserted yet.
    *                +----------+............+
    *   added to result-----^        ^------ not added to result (processed on next iterations)
    */

   MemoryChunk::intersect( leftChunk, rightChunk, rightLeftOver );

   //add the right chunk 
   //hint = this->insert( hint, typename BaseMap::value_type( rightChunk, NEW _Type( *hint->second ) ) );
   //result: right chunk
   ptrList.push_back( MemChunkPair ( NEW MemoryChunk( rightChunk ), &(hint->second) ) );
   //leftover not added to result since more overlapping may exist
   key = rightLeftOver;  //try to add what is left
}

template < typename _Type >
void MemoryMap< _Type >::processTotalOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   //_Type *ptr = (_Type *) NULL;
   //message0("TOTAL_OVERLAP detected " << (void *) iterKey.getAddress() << " : " << iterKey.getLength() << " versus " << (void *) hint->first.getAddress() << " : " << hint->first.getLength() );
   MemoryChunk leftChunk = key;
   MemoryChunk rightChunk = hint->first ;
   MemoryChunk leftLeftOver;
   /*
    *                +=====================+
    *                |     rightChunk      |
    *                |  (already inserted) |
    *                +=====================+
    *   +-----------------------------------------------+
    *   |                  leftChunk                    |
    *   |               (to be inserted)                |
    *   +-----------------------------------------------+
    *                    == partition ==
    *                +=====================+
    *                |     rightChunk      |
    *                +=====================+
    *   +------------+        ^            +............+
    *   | leftChunk  |        |            |leftLeftOver|
    *   +------------+        |            +............+
    *         ^------  added to result          ^-- not added (keep processing)
    */

   MemoryChunk::partition( leftChunk, rightChunk, leftLeftOver );

   //hint = this->insert( hint, typename BaseMap::value_type( leftChunk, ptr ) ); //add the left chunk, 
   ptrList.push_back( MemChunkPair ( NEW MemoryChunk ( leftChunk ), NULL ) ); //return: leftChunk
   ptrList.push_back( MemChunkPair ( NEW MemoryChunk ( rightChunk ), &(hint->second) ) ); //return: rightChunk
   key = leftLeftOver; //try to add what is left
}

template < typename _Type >
void MemoryMap< _Type >::processSubchunkOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   MemoryChunk leftChunk = hint->first ;
   MemoryChunk rightChunk = key;
   MemoryChunk leftLeftOver;
   /*
    *   +===============================================+
    *   |                  leftChunk                    |
    *   |               (already inserted)              |
    *   +===============================================+
    *                +---------------------+
    *                |     rightChunk      |
    *                |  (to be inserted)   |
    *                +---------------------+
    *                    == partition ==
    *   +============+                     +............+
    *   | leftChunk  |                     |leftLeftOver|
    *   +============+                     +............+
    *                +---------------------+
    *                |     rightChunk      |
    *                +---------------------+
    *                   added to result
    */

   MemoryChunk::partition( leftChunk, rightChunk, leftLeftOver );

   //hint = this->insert( hint, typename BaseMap::value_type( rightChunk, NEW _Type( *hint->second ) ) ); //add the right chunk, 
   ptrList.push_back( MemChunkPair ( NEW MemoryChunk( rightChunk ), &(hint->second) ) ); //return: rightChunk
   //hint = this->insert( hint, typename BaseMap::value_type( leftLeftOver, NEW _Type( *hint->second ) ) ); //add the leftover
}

template < typename _Type >
void MemoryMap< _Type >::processTotalBeginOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   MemoryChunk leftChunk = hint->first ;
   MemoryChunk rightChunk = key;
   /*
    *   +=====================+
    *   |      leftChunk      |
    *   |  (already inserted) |
    *   +=====================+
    *   +-----------------------------------------------+
    *   |                 rightChunk                    |
    *   |               (to be inserted)                |
    *   +-----------------------------------------------+
    *                    == partitionBeginAltB ==
    *   +=====================+
    *   |      leftChunk      |
    *   +=====================+
    *             ^           ...........................
    *             |           :       rightChunk        :
    *             |           :.........................:
    *       added to result         ^-- not added (keep processing)
    */
   MemoryChunk::partitionBeginAltB( leftChunk, rightChunk );
   ptrList.push_back( MemChunkPair ( NEW MemoryChunk ( rightChunk ), &(hint->second) ) ); //return: leftChunk
   key = rightChunk;
}

template < typename _Type >
void MemoryMap< _Type >::processSubchunkBeginOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   MemoryChunk leftChunk = hint->first ;
   MemoryChunk rightChunk = key;
   /*
    *   +===============================================+
    *   |                  leftChunk                    |
    *   |               (already inserted)              |
    *   +===============================================+
    *   +---------------------+
    *   |     rightChunk      |
    *   |  (to be inserted)   |
    *   +---------------------+
    *                == partitionBeginAgtB ==
    *                         +-------------------------+
    *                         |        rightChunk       |
    *                         +-------------------------+
    *   +=====================+
    *   |     leftchunk       |
    *   +=====================+
    *       added to result
    */
   MemoryChunk::partitionBeginAgtB( leftChunk, rightChunk );
   ptrList.push_back( MemChunkPair ( NEW MemoryChunk( leftChunk ), &(hint->second) ) ); //return: leftChunk
   //hint = this->insert( hint, typename BaseMap::value_type( rightChunk, NEW _Type( *hint->second ) ) ); //add the rightChunk
}

template < typename _Type >
void MemoryMap< _Type >::processTotalEndOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   //_Type *ptr = (_Type *) NULL;
   MemoryChunk leftChunk = key;
   MemoryChunk rightChunk = hint->first ;
   //const MemoryChunk *rightChunkPtr = &( hint->first );
   _Type ** rightChunkDataPtr = &( hint->second );
   /*
    *                             +=====================+
    *                             |     rightChunk      |
    *                             |  (already inserted) |
    *                             +=====================+
    *   +-----------------------------------------------+
    *   |                  leftChunk                    |
    *   |               (to be inserted)                |
    *   +-----------------------------------------------+
    *                    == partition ==
    *                             +=====================+
    *                             |     rightChunk      |
    *                             +=====================+
    *   +-------------------------+        ^
    *   |        leftChunk        |        |
    *   +-------------------------+        |
    *         ^------  added to result ----+
    */
   MemoryChunk::partitionEnd( leftChunk, rightChunk );
   //hint = this->insert( hint, typename BaseMap::value_type( rightChunk, ptr ) ); //add the leftChunk
   ptrList.push_back( MemChunkPair ( NEW MemoryChunk ( leftChunk ), NULL ) ); //return: leftChunk
   ptrList.push_back( MemChunkPair ( NEW MemoryChunk ( rightChunk ), rightChunkDataPtr ) ); //return: rightChunk
}

template < typename _Type >
void MemoryMap< _Type >::processSubchunkEndOverlapNI( MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   MemoryChunk leftChunk = hint->first ;
   MemoryChunk rightChunk = key;
   /*
    *   +===============================================+
    *   |                  leftChunk                    |
    *   |               (already inserted)              |
    *   +===============================================+
    *                             +---------------------+
    *                             |     rightChunk      |
    *                             |  (to be inserted)   |
    *                             +---------------------+
    *                    == partition ==
    *   +=========================+
    *   |        leftChunk        |
    *   +=========================+
    *                             +---------------------+
    *                             |     rightChunk      |
    *                             +---------------------+
    *                                added to result
    */
   MemoryChunk::partitionEnd( leftChunk, rightChunk );
   //hint = this->insert( hint, typename BaseMap::value_type( rightChunk, NEW _Type( *hint->second ) ) ); //add the rightChunk
   ptrList.push_back( MemChunkPair ( NEW MemoryChunk ( rightChunk ), &(hint->second) ) ); //return: rightChunk
}

template < typename _Type >
void MemoryMap< _Type >::insertWithOverlap( const MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   // precondition: there is no exact match
   bool lastChunk = false;
   MemoryChunk iterKey = key;
   if ( !this->empty() )
   {
      if ( hint != this->begin() )
         hint--;
      do {
         switch ( hint->first.checkOverlap( iterKey ) )
         {
            case MemoryChunk::NO_OVERLAP:
               processNoOverlap( iterKey, hint, ptrList );
               lastChunk = true;
               break;
            case MemoryChunk::BEGIN_OVERLAP:
               processBeginOverlap( iterKey, hint, ptrList );
               lastChunk = true; //this is a final situation since the to-be-inserted chunk ends.
               break;
            case MemoryChunk::END_OVERLAP:
               processEndOverlap( iterKey, hint, ptrList );
               break;
            case MemoryChunk::TOTAL_OVERLAP:
               processTotalOverlap( iterKey, hint, ptrList );
               break;
            case MemoryChunk::SUBCHUNK_OVERLAP:
               processSubchunkOverlap( iterKey, hint, ptrList );
               lastChunk = true;
               break;
            case MemoryChunk::TOTAL_BEGIN_OVERLAP:
               processTotalBeginOverlap( iterKey, hint, ptrList );
               break;
            case MemoryChunk::SUBCHUNK_BEGIN_OVERLAP:
               processSubchunkBeginOverlap( iterKey, hint, ptrList );
               lastChunk = true;
               break;
            case MemoryChunk::TOTAL_END_OVERLAP:
               processTotalEndOverlap( iterKey, hint, ptrList );
               lastChunk = true;
               break;
            case MemoryChunk::SUBCHUNK_END_OVERLAP:
               processSubchunkEndOverlap( iterKey, hint, ptrList );
               lastChunk = true;
               break;
         }
         hint++;
      } while ( hint != this->end() && !lastChunk );
   } else {
      hint = this->insert( hint, typename BaseMap::value_type( key, ( _Type * ) NULL ) );
      ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) );
   }
}

template < typename _Type >
void MemoryMap< _Type >::getWithOverlap( const MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   // precondition: there is no exact match
   bool lastChunk = false;
   MemoryChunk iterKey = key;
   if ( !this->empty() )
   {
      if ( hint != this->begin() )
         hint--;
      do {
         switch ( hint->first.checkOverlap( iterKey ) )
         {
            case MemoryChunk::NO_OVERLAP:
               processNoOverlapNI( iterKey, hint, ptrList );
               lastChunk = true;
               break;
            case MemoryChunk::BEGIN_OVERLAP:
               processBeginOverlapNI( iterKey, hint, ptrList );
               lastChunk = true; //this is a final situation since the to-be-inserted chunk ends.
               break;
            case MemoryChunk::END_OVERLAP:
               processEndOverlapNI( iterKey, hint, ptrList );
               break;
            case MemoryChunk::TOTAL_OVERLAP:
               processTotalOverlapNI( iterKey, hint, ptrList );
               break;
            case MemoryChunk::SUBCHUNK_OVERLAP:
               processSubchunkOverlapNI( iterKey, hint, ptrList );
               lastChunk = true;
               break;
            case MemoryChunk::TOTAL_BEGIN_OVERLAP:
               processTotalBeginOverlapNI( iterKey, hint, ptrList );
               break;
            case MemoryChunk::SUBCHUNK_BEGIN_OVERLAP:
               processSubchunkBeginOverlapNI( iterKey, hint, ptrList );
               lastChunk = true;
               break;
            case MemoryChunk::TOTAL_END_OVERLAP:
               processTotalEndOverlapNI( iterKey, hint, ptrList );
               lastChunk = true;
               break;
            case MemoryChunk::SUBCHUNK_END_OVERLAP:
               processSubchunkEndOverlapNI( iterKey, hint, ptrList );
               lastChunk = true;
               break;
         }
         hint++;
      } while ( hint != this->end() && !lastChunk );
   } else {
      ptrList.push_back( MemChunkPair ( NEW MemoryChunk(key), NULL ) );
   }
}

template < typename _Type >
void MemoryMap< _Type >::getOrAddChunk( uint64_t addr, std::size_t len, MemChunkList &resultEntries )
{
   MemoryChunk key( addr, len );

   typename BaseMap::iterator it = this->lower_bound( key );
   if ( it == this->end() || this->key_comp()( key, it->first ) || it->first.getLength() != len )
   {
      /* NOT EXACT ADDR FOUND: "addr" is higher than any odther addr in the map OR less than "it" */
      insertWithOverlap( key, it, resultEntries );
      //res = MEM_CHUNK_NOT_FOUND_BUT_ALLOCATED;
   }
   else
   {
      /* EXACT ADDR FOUND */
      resultEntries.push_back( MemChunkPair( &(it->first), &(it->second) ) );
   }
}

template < typename _Type >
void MemoryMap< _Type >::getChunk2( uint64_t addr, std::size_t len, MemChunkList &resultEntries )
{
   MemoryChunk key( addr, len );

   typename BaseMap::iterator it = this->lower_bound( key );
   if ( it == this->end() || this->key_comp()( key, it->first ) || it->first.getLength() != len )
   {
      /* NOT EXACT ADDR FOUND: "addr" is higher than any odther addr in the map OR less than "it" */
      getWithOverlap( key, it, resultEntries );
   }
   else
   {
      /* EXACT ADDR FOUND */
      resultEntries.push_back( MemChunkPair( NEW MemoryChunk( key ), &(it->second) ) );
   }
}

template < typename _Type >
void MemoryMap< _Type >::print() const
{
   int i = 0;
   typename BaseMap::const_iterator it = this->begin();
   std::cerr << "printing memory chunks" << std::endl;
   for (; it != this->end(); it++)
   {
      std::cerr << "\tchunk: " << i++ << " addr=" << (void *) it->first.getAddress() <<"(" << (uint64_t)it->first.getAddress()<< ")" << " len=" << it->first.getLength() << " ptr val is " << it->second << " addr of ptr val is " << (void *) &(it->second) << std::endl;
      it->second->print();
   } 
   std::cerr << "end of memory chunks" << std::endl;
}

/* merge input MemoryMap mm into this */
template < typename _Type >
void MemoryMap< _Type >::merge( const MemoryMap< _Type > &mm )
{
   typename BaseMap::const_iterator inputIt = mm.begin();
   
   while ( inputIt != mm.end() )
   {
      MemChunkList result;
      typename MemChunkList::iterator resultIt;
      getOrAddChunk( inputIt->first.getAddress(), inputIt->first.getLength(), result);
      for ( resultIt = result.begin(); resultIt != result.end(); resultIt++ )
      {
         _Type **ent = resultIt->second;
         if ( *ent == NULL ) { *ent = NEW _Type ( *inputIt->second ); /*if ( sys.getNetwork()->getNodeNum()==0){fprintf(stderr, "[%p] New entry from merge tag %p\n", this, (void *)inputIt->first.getAddress() );}*/ }
         else { /*if ( sys.getNetwork()->getNodeNum()==0){ fprintf(stderr, "[%p] Already got that entry %p\n", this, (void *) inputIt->first.getAddress()); (*ent)->print(); inputIt->second->print();}*/ (*ent)->merge( *(inputIt->second) );}
      }
      inputIt++;
   }
}

template < typename _Type >
void MemoryMap< _Type >::expand( MemoryChunk &inputKey, _Type *inputData, typename BaseMap::iterator &thisIt )
{
   typename BaseMap::iterator nextIt = thisIt;
   nextIt++;
   MemoryChunk thisKey = thisIt->first;
   _Type *thisData = thisIt->second;
   bool thisAndInputDataAreEqual = thisData->equal( *inputData );
            //fprintf(stderr, "entry %s, this %d-%d  input %d-%d\n", __FUNCTION__, thisKey.getAddress(), thisKey.getLength(), inputKey.getAddress(), inputKey.getLength() );

   if ( nextIt == this->end() )
   {
      if ( thisAndInputDataAreEqual ) {
         /* we can expand safetly, and we finish */
         MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
         thisNoConst.expandIncluding( inputKey );
      } else {
         thisIt = this->insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type ( *inputData ) ) );
      }
      thisIt++;
   } else {
      MemoryChunk nextKey = nextIt->first;
      _Type *nextData = nextIt->second;
      bool nextAndInputDataAreEqual = nextData->equal( *inputData );


      /* check overlap with the next item in the map! */
      switch ( nextKey.checkOverlap( inputKey ) ) {
         case MemoryChunk::NO_OVERLAP:
       //     fprintf(stderr, "case NO_OVERLAP %s\n", __FUNCTION__ );
            if ( thisAndInputDataAreEqual ) {
               /* we can expand safetly, and we finish */
               MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
               thisNoConst.expandIncluding( inputKey );
            } else {
               thisIt = this->insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type ( *inputData ) ) );
            }
            thisIt++;
            break;
         case MemoryChunk::BEGIN_OVERLAP:
            //fprintf(stderr, "case BEGIN_OVERLAP %s\n", __FUNCTION__ );
            if ( thisAndInputDataAreEqual ) {
               /* if the chunk Data is equivalent, we can expan all chunks and we finish */
               if ( nextAndInputDataAreEqual )
               {
                  /* expand to fit [this ... inputKey ... next]
                   * then erase next entry
                   */
                  MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
                  thisNoConst.expandIncluding( nextKey );
                  delete nextIt->second;
                  this->erase( nextIt );
                  thisIt = this->find( thisKey );
               } else {
                  /* fit [this ... inputKey ... ]{next:overlap_with_input}(next:leftover)
                   */
                  MemoryChunk rightLeftOver;
                  MemoryChunk inputKeyCopy = inputKey;
                  MemoryChunk &rightChunk = const_cast< MemoryChunk & >( nextIt->first );
                  /* we can not merge both already-inserted entries,
                   * expand "this" as much as possible and partition
                   * the leftover of "inputKey" that intersects with hintCopy
                   */
                  MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
                  thisNoConst.expandExcluding( nextIt->first );
                  MemoryChunk::intersect( inputKeyCopy, rightChunk, rightLeftOver ); //this modifies nextIt->first!!
                  thisIt = this->insert( nextIt, typename BaseMap::value_type( rightLeftOver, NEW _Type( *nextIt->second ) ) );
                  thisIt--;
                  thisIt->second->merge( *inputData );
                  thisIt++;
               }
            } else {
               if ( nextAndInputDataAreEqual )
               {
                  /* expand to fit: (this)[... inputKey ... next]
                   */
                  std::pair<typename BaseMap::iterator, bool> insertResult;
                  inputKey.expandIncluding( nextKey );
                  _Type *data = nextIt->second;
                  this->erase( nextIt );
                  insertResult = this->insert( typename BaseMap::value_type( inputKey, data ) ); //reuse data entry
                  thisIt = insertResult.first;
               } else {
                  /* fit (this)[ ... inputKey ... ]{next:overlap_with_input}(next:leftover)
                   */
                  MemoryChunk rightLeftOver;
                  MemoryChunk inputKeyCopy = inputKey;
                  MemoryChunk &rightChunk = const_cast< MemoryChunk & >( nextIt->first );
                  /* we can not merge both already-inserted entries,
                   * expand "this" as much as possible and partition
                   * the leftover of "inputKey" that intersects with hintCopy
                   */
                  MemoryChunk::intersect( inputKeyCopy, rightChunk, rightLeftOver ); //this modifies nextIt->first!!
                  thisIt = this->insert( thisIt, typename BaseMap::value_type( inputKeyCopy, NEW _Type( *inputData ) ) );
                  thisIt++;
                  thisIt = this->insert( thisIt, typename BaseMap::value_type( rightLeftOver, NEW _Type( *thisIt->second ) ) );
                  thisIt--;
                  thisIt->second->merge( *inputData ); //merge overlapped chunk
                  thisIt++;
               }
            }
            break;
         case MemoryChunk::END_OVERLAP:
            /* ERROR, it is impossible to have End Overlap with hintNext */
            //fprintf(stderr, "error END_OVERLAP %s\n", __FUNCTION__ );
            break;
         case MemoryChunk::TOTAL_OVERLAP:
            //fprintf(stderr, "case TOTAL_OVERLAP %s\n", __FUNCTION__ );
            if ( thisAndInputDataAreEqual ) {
               MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
               if ( nextAndInputDataAreEqual )
               {
                  /* [this .. inputKey .. next]{..inputKey..} */
                  thisNoConst.expandIncluding( nextKey );
                  MemoryChunk key = thisIt->first;
                  delete nextIt->second;
                  this->erase( nextIt );
                  thisIt = nextIt = this->find( key );
                  nextIt++;
                  inputKey.cutAfter( thisIt->first );
                  /* try expanding from this new state */
                  this->expand( inputKey, inputData, thisIt );
               } else {
                  /* [this .. inputKey .. ][next]{..inputKey..} */
                  thisNoConst.expandExcluding( nextKey );
                  nextIt->second->merge( *inputData );
                  inputKey.cutAfter( nextKey );
                  /* try expanding from this new state */
                  this->expand( inputKey, inputData, nextIt );
               }
            } else {
               if ( nextAndInputDataAreEqual )
               {
                  /* (this)[.. inputKey .. next]{..inputKey..} */
                  std::pair<typename BaseMap::iterator, bool> insertResult;
                  MemoryChunk key = inputKey;
                  key.expandIncluding( nextKey );
                  _Type *data = nextIt->second;
                  this->erase( nextIt );
                  insertResult = this->insert( typename BaseMap::value_type( inputKey, data ) ); //reuse data entry
                  thisIt = insertResult.first;
                  inputKey.cutAfter( thisIt->first );
                  /* try expanding from this new state */
                  this->expand( inputKey, inputData, thisIt );
               } else {
                  /* (this)[.. inputKey .. ][next]{inputKey} */
                  MemoryChunk key = inputKey;
                  key.expandExcluding( nextKey );
                  nextIt->second->merge( *inputData );
                  inputKey.cutAfter( nextKey );
                  thisIt = this->insert( thisIt, typename BaseMap::value_type( key, NEW _Type( *inputData ) ) );
                  thisIt++;
                  /* try expanding from this new state */
                  this->expand( inputKey, inputData, thisIt );
               }
            }
            break;
         case MemoryChunk::SUBCHUNK_OVERLAP:
            /* ERROR, it is impossible to have Subchunk Overlap with hintNext */
            fprintf(stderr, "error SUBCHUNK_OVERLAP %s\n", __FUNCTION__ );
            break;
         case MemoryChunk::TOTAL_BEGIN_OVERLAP:
            //fprintf(stderr, "case TOTAL_BEGIN_OVERLAP %s\n", __FUNCTION__ );
            if ( thisAndInputDataAreEqual ) {
               if ( nextAndInputDataAreEqual ) {
                  /* expand to fit [this][next]{next:leftover}
                   * then erase next entry
                   */
                  MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
                  thisNoConst.expandIncluding( nextKey );
                  inputKey.cutAfter( nextKey );
                  delete nextIt->second;
                  this->erase( nextIt );
                  thisIt = this->find( thisKey );
                  this->expand( inputKey, inputData, nextIt );
               } else {
                  /* fit (this){next:partition_with_input}(next:leftover)
                   * merge partition, continue processing leftover
                   */
                  MemoryChunk inputKeyCopy = inputKey;
                  MemoryChunk &rightChunk = const_cast< MemoryChunk & >( nextIt->first );
                  /* we can not merge both already-inserted entries,
                   * expand "this" as much as possible and partition
                   * the leftover of "inputKey" that intersects with hintCopy
                   */
                  MemoryChunk::partitionBeginAltB( nextKey, inputKey );
                  nextIt->second->merge( *inputData );
                  this->expand( inputKey, inputData, nextIt );
               }
            } else {
               if ( nextAndInputDataAreEqual ) {
                  /* expand to fit (this)[next]{next:leftover}
                   * since next and input are equal, no action is required,
                   * just adjust the input key and call recursively to keep
                   * processing if there are more chunks that can be merged.
                   */
                  inputKey.cutAfter( nextKey );
                  this->expand( inputKey, inputData, nextIt);
               } else {
                  /* expand to fit (this)[next]{next:leftover}
                   * similar than before but we need to merge the information
                   * of "next" and "input".
                   */
                  nextIt->second->merge( *inputData );
                  inputKey.cutAfter( nextKey );
                  this->expand( inputKey, inputData, nextIt);
               }
            }
            break;
         case MemoryChunk::SUBCHUNK_BEGIN_OVERLAP:
            //fprintf(stderr, "error SUBCHUNK_BEGIN_OVERLAP %s\n", __FUNCTION__ );
            if ( thisAndInputDataAreEqual ) {
               if ( nextAndInputDataAreEqual ) {
                  MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
                  thisNoConst.expandIncluding( nextKey );
                  this->erase( nextIt );
               } else {
                  MemoryChunk &nextNoConst = const_cast< MemoryChunk & >( nextIt->first );
                  nextNoConst = inputKey;
                  nextKey.cutAfter( inputKey );
                  thisIt = this->insert( nextIt, typename BaseMap::value_type( nextKey, NEW _Type ( *nextData ) ) );
                  nextData->merge( *inputData ); //merge after insert to create a copy of the original "next" entry.
               }
            } else {
               if ( nextAndInputDataAreEqual ) {
               //no expand with "this" and "next" already ok, do nothing
               } else {
                  _Type * tmpData = NEW _Type ( *inputData );
                  tmpData->merge( *nextData );
                  if ( tmpData->equal( *thisData ) ) {
                     std::pair<typename BaseMap::iterator, bool> insertResult;
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
                     thisNoConst.expandIncluding( inputKey );
                     this->erase( nextIt );
                     insertResult = this->insert( typename BaseMap::value_type( nextKey, nextData ) ); //reuse entry
                     delete tmpData;
                  } else {
                     MemoryChunk &nextNoConst = const_cast< MemoryChunk & >( nextIt->first );
                     nextNoConst = inputKey;
                     nextKey.cutAfter( inputKey );
                     thisIt = this->insert( nextIt, typename BaseMap::value_type( nextKey, tmpData ) );
                  }
               }
            }
            break;
         case MemoryChunk::TOTAL_END_OVERLAP:
            //fprintf(stderr, "case TOTAL_END_OVERLAP %s\n", __FUNCTION__ );
            if ( thisAndInputDataAreEqual ) {
               MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
               if ( nextAndInputDataAreEqual ) {
                  thisNoConst.expandIncluding( nextKey );
                  delete nextIt->second;
                  this->erase( nextIt );
               } else {
                  thisNoConst.expandExcluding( nextKey );
                  nextIt->second->merge( *inputData );
               }
               thisIt++;
            } else {
               if ( nextAndInputDataAreEqual ) {
                  thisIt = this->insert( thisIt, typename BaseMap::value_type( inputKey, nextIt->second ) );
                  thisIt++;
                  delete thisIt->second;
                  this->erase( thisIt );
                  thisIt = this->find( inputKey );
               } else {
                  inputKey.expandExcluding( nextKey );
                  nextIt->second->merge( *inputData );
                  thisIt = this->insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type ( *inputData ) ) );
               }
               thisIt++;
            }
            break;
         case MemoryChunk::SUBCHUNK_END_OVERLAP:
            /* ERROR, it is impossible to have Subchunk End Overlap with hintNext */
            fprintf(stderr, "error SUBCHUNK_END_OVERLAP %s\n", __FUNCTION__ );
            break;
      }
   }
   //fprintf(stderr, "exit %s\n", __FUNCTION__ );
}

/* merge input MemoryMap mm into this */
template < typename _Type >
void MemoryMap< _Type >::merge2( const MemoryMap< _Type > &mm )
{
   typename BaseMap::const_iterator inputIt = mm.begin();
   typename BaseMap::iterator thisIt = this->begin();
   typename BaseMap::iterator insertIt;

            //fprintf(stderr, "ENTRY %s\n", __FUNCTION__ );
   while ( inputIt != mm.end() )
   {
      if ( thisIt == this->end() ) {
         //insert the rest of the input MemoryMap, because there are no more entries in "this"
         for (; inputIt != mm.end(); inputIt++ )
         {
            thisIt = this->insert( thisIt, typename BaseMap::value_type( inputIt->first, NEW _Type ( *( inputIt->second ) ) ) );
            //FIXME: check for merge oportunities between inserted chunks!
         }
      } else {
         //fprintf(stderr, "eq? check ovrl %s %p-%d vs %p-%d\n", __FUNCTION__ , thisKey.getAddress(), thisKey.getLength(), inputKey.getAddress(), inputKey.getLength() );
         MemoryChunk thisKey = thisIt->first, inputKey = inputIt->first;
         _Type *thisData = thisIt->second, *inputData = inputIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( thisKey.equal( inputKey ) ) {
            if ( !thisAndInputDataAreEqual )
            {
               thisIt->second->merge( *inputData );
            }
            inputIt++;
            thisIt++;
         } else {
            MemoryChunk leftLeftOver;
            switch ( thisKey.checkOverlap( inputKey ) )
            {
               case MemoryChunk::NO_OVERLAP:
                  //std::cerr << (void *)thisKey.getAddress() << " vs " <<(void*)inputKey.getAddress() << std::endl;
                  if ( thisKey.getAddress() + thisKey.getLength() == inputKey.getAddress() && thisAndInputDataAreEqual )
                  {
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
                     thisNoConst.expandIncluding( inputKey );
                     typename BaseMap::iterator nextIt = thisIt;
                     nextIt++;
                     if ( thisNoConst.getAddress() + thisNoConst.getLength() == nextIt->first.getAddress() && thisData->equal( *( nextIt->second ) ) ) {
                        thisNoConst.expandIncluding( nextIt->first );
                        delete nextIt->second;
                        this->erase( nextIt );
                        thisIt = this->find( thisNoConst );
                     }
                     inputIt++;
                  } else if ( inputKey.getAddress() + inputKey.getLength() == thisKey.getAddress() && thisAndInputDataAreEqual ) {
                     std::pair<typename BaseMap::iterator, bool> insertResult;
                     inputKey.expandIncluding( thisKey );
                     this->erase( thisIt );
                     insertResult = this->insert( typename BaseMap::value_type( inputKey, thisData ) ); //reuse data
                     thisIt = insertResult.first;
                  } else if ( thisKey.getAddress() > inputKey.getAddress() ) {
                     thisIt = this->insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *inputData ) ) );
                     thisIt++;
                     inputIt++;
                  } else { thisIt++; }
                  break;
               case MemoryChunk::BEGIN_OVERLAP:
                  if ( thisAndInputDataAreEqual ) {
                     /* I CAN MERGE */
                     std::pair<typename BaseMap::iterator, bool> insertResult;
                     inputKey.expandIncluding( thisKey );
                     this->erase( thisIt );
                     insertResult = this->insert( typename BaseMap::value_type( inputKey, thisData ) ); //reuse data
                     // insertResult.second must be true
                     thisIt = insertResult.first;
                  } else {
                     MemoryChunk rightLeftOver;
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
                     MemoryChunk::intersect( inputKey, thisNoConst, rightLeftOver );
                     thisIt = this->insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *inputData ) ));
                     thisIt = this->insert( thisIt, typename BaseMap::value_type( rightLeftOver, NEW _Type( *thisData ) ));
                     thisData->merge( *inputData );
                  }
                  inputIt++;
                  break;
               case MemoryChunk::END_OVERLAP:
                  if ( thisAndInputDataAreEqual ) {
                     MemoryChunk rightLeftOver;
                     MemoryChunk::intersect( thisKey, inputKey, rightLeftOver );
                     this->expand( rightLeftOver, inputData, thisIt );
                  }
                  else {
                     MemoryChunk rightLeftOver;
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
                     MemoryChunk::intersect( thisNoConst, inputKey, rightLeftOver );
                     thisIt = this->insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *thisData ) ));
                     thisIt->second->merge( *inputData );
                     this->expand( rightLeftOver, inputData, thisIt );
                  }
                  inputIt++;
                  break;
               case MemoryChunk::TOTAL_OVERLAP:
                  MemoryChunk::partition( inputKey, thisKey, leftLeftOver );
                  if ( thisAndInputDataAreEqual ) {
                     std::pair<typename BaseMap::iterator, bool> insertResult;
                     inputKey.expandIncluding( thisKey );
                     this->erase( thisIt );
                     insertResult = this->insert( typename BaseMap::value_type( inputKey, thisData ) ); // do not allocate new data, reuse old
                     thisIt = insertResult.first;
                  } else {
                     thisIt->second->merge( *inputData );
                     thisIt = this->insert( thisIt, typename BaseMap::value_type( inputKey, inputData ));
                     thisIt++;
                  }
                  this->expand( leftLeftOver, inputData, thisIt );
                  inputIt++;
                  break;
               case MemoryChunk::SUBCHUNK_OVERLAP:
                  if ( thisAndInputDataAreEqual ) {
                     //do nothing
                  } else {
                     _Type *data = thisIt->second;
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
                     MemoryChunk::partition( thisNoConst, inputKey, leftLeftOver );
                     thisIt = this->insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *inputIt->second ) ) );
                     thisIt = this->insert( thisIt, typename BaseMap::value_type( leftLeftOver, NEW _Type( *data ) ) );
                  }
                  inputIt++;
                  break;
               case MemoryChunk::TOTAL_BEGIN_OVERLAP:
                  MemoryChunk::partitionBeginAltB( thisKey, inputKey );
                  if ( thisAndInputDataAreEqual ) { //do nothing
                  } else {
                     thisIt->second->merge( *inputData );
                  }
                  this->expand( inputKey, inputData, thisIt );
                  inputIt++;
                  break;
               case MemoryChunk::SUBCHUNK_BEGIN_OVERLAP:
                  if ( thisAndInputDataAreEqual ) {
                     //do_nothing
                  } else {
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
                     MemoryChunk::partitionBeginAgtB( thisNoConst, inputKey );
                     thisIt = this->insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *thisData ) ) );
                     thisData->merge( *inputData );
                  }
                  inputIt++;
                  break;
               case MemoryChunk::TOTAL_END_OVERLAP:
            //fprintf(stderr, "case TOTAL_END_OVERLAP %s\n", __FUNCTION__ );
                  if ( thisAndInputDataAreEqual ) {
                     std::pair<typename BaseMap::iterator, bool> insertResult;
                     this->erase( thisIt );
                     insertResult = this->insert( typename BaseMap::value_type( inputKey, thisData ) ); //reuse data
                     thisIt = insertResult.first;
                  } else {
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
                     MemoryChunk::partitionEnd( inputKey, thisNoConst );
                     thisIt = this->insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type ( *inputData ) ) );
                     thisIt++;
                  }
                  thisIt++;
                  inputIt++;
                  break;
               case MemoryChunk::SUBCHUNK_END_OVERLAP:
                  if ( thisAndInputDataAreEqual ) {
                     //do_nothing;
                  } else {
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
                     MemoryChunk::partitionEnd( thisNoConst, inputKey );
                     thisIt = this->insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *inputData ) ) );
                     thisIt->second->merge( *thisData );
                  }
                  thisIt++;
                  inputIt++;
                  break;
            }
         }
      }
   }
}
}

#endif /* MEMORYMAP_H */
