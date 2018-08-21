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

#ifndef MEMORYMAP_H
#define MEMORYMAP_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "memorymap_decl.hpp"
#include "basethread_decl.hpp"
#include "debug.hpp"

namespace nanos {

inline MemoryChunk& MemoryChunk::operator=( MemoryChunk const &mc ) {
   _addr = mc._addr;
   _len = mc._len;
   return *this;
}

inline bool MemoryChunk::operator<( MemoryChunk const &chunk ) const {
   return _addr < chunk._addr;
} 

inline uint64_t MemoryChunk::getAddress() const {
   return _addr;
}
inline std::size_t MemoryChunk::getLength() const {
   return _len;
}

inline bool MemoryChunk::equal( MemoryChunk const &target ) const {
   return ( _addr == target._addr && _len == target._len );
}

inline bool MemoryChunk::contains( MemoryChunk const &target ) const {
   return ( ( _addr <= target._addr) && ( ( _addr + _len ) >= ( target._addr + target._len ) ) );
}

inline MemoryChunk::OverlapType MemoryChunk::checkOverlap( MemoryChunk const &target ) const {
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
      //fprintf( stderr, "detected %s\n", strOverlap[ret] );
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

inline void MemoryChunk::partition( MemoryChunk &mcA, MemoryChunk const &mcB, MemoryChunk &mcC ) {
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

inline void MemoryChunk::partitionBeginAltB( MemoryChunk const &mcA, MemoryChunk &mcB ) {
   //assume A.addr = B.addr, A.len < B.len,  A.addr is NOT modified 
   std::size_t bLen = mcB._len - mcA._len;
   mcB._addr = mcA._addr + mcA._len;
   mcB._len = bLen;
}

inline void MemoryChunk::partitionEnd( MemoryChunk &mcA, MemoryChunk const &mcB ) {
   //assume A.addr+A.len = B.addr+B.len, A.len > B.len,  B.addr is NOT modified 
   mcA._len -= mcB._len;
}

inline void MemoryChunk::expandIncluding( MemoryChunk const &mcB ) {
   _len = ( mcB._addr + mcB._len ) - _addr;
}

inline void MemoryChunk::expandExcluding( MemoryChunk const &mcB ) {
   _len = ( mcB._addr - _addr );
}

inline void MemoryChunk::cutAfter( MemoryChunk const &mc ) {
   _len = ( _addr + _len ) - ( mc._addr + mc._len );
   _addr = mc._addr + mc._len;
}


template < typename _Type >
void MemoryMap< _Type >::insertWithOverlap( const MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   class LocalFunctions {
      MemoryMap< _Type > &_thisMap;
      MemoryChunk        &_key;
      iterator           &_hint;
      MemChunkList       &_ptrList;

      public:
      LocalFunctions( MemoryMap< _Type > &thisMap, MemoryChunk &localKey, iterator &localHint, MemChunkList &localPtrList ) :
         _thisMap( thisMap ), _key( localKey ), _hint( localHint ), _ptrList( localPtrList ) { }

      void insertNoOverlap()
      {
         _Type *ptr = (_Type *) NULL;
         ensure( _key.getLength() > 0, "Invalid lengtgh." );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( _key, ptr ) );
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) );
      }

      void insertBeginOverlap()
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
         MemoryChunk &leftChunk = _key;
         MemoryChunk &rightChunk = const_cast< MemoryChunk & >( _hint->first );
         MemoryChunk rightLeftOver;

         MemoryChunk::intersect( leftChunk, rightChunk, rightLeftOver );

         ensure( leftChunk.getLength() > 0, "Invalid lengtgh." );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( leftChunk, ptr ) ); //add the left chunk, 
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) );
         _hint++; //advance the iterator (it should reach "rightChunk"!)
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) );

         ensure( rightLeftOver.getLength() > 0, "Invalid lengtgh." );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightLeftOver, NEW _Type( *_hint->second ) ) );
      }

      void insertEndOverlap()
      {
         MemoryChunk &leftChunk = const_cast< MemoryChunk & >( _hint->first );
         MemoryChunk &rightChunk = _key;
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
         ensure( rightChunk.getLength() > 0, "Invalid lengtgh." );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightChunk, NEW _Type( *_hint->second ) ) );
         //result: right chunk
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second) ) );
         //leftover not added to result since more overlapping may exist
         _key = rightLeftOver;  //try to add what is left
      }

      void insertTotalOverlap()
      {
         _Type *ptr = (_Type *) NULL;
         MemoryChunk &leftChunk = _key;
         MemoryChunk &rightChunk = const_cast< MemoryChunk & >( _hint->first );
         _Type ** rightChunkDataPtr = &( _hint->second );
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

         ensure( leftChunk.getLength() > 0, "Invalid lengtgh." );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( leftChunk, ptr ) ); //add the left chunk, 
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: leftChunk
         _ptrList.push_back( MemChunkPair ( &rightChunk, rightChunkDataPtr ) ); //return: rightChunk
         _key = leftLeftOver; //try to add what is left
      }

      void insertSubchunkOverlap()
      {
         MemoryChunk &leftChunk = const_cast< MemoryChunk & >( _hint->first );
         MemoryChunk &rightChunk = _key;
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

         ensure( rightChunk.getLength() > 0, "Invalid lengtgh." );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightChunk, NEW _Type( *_hint->second ) ) ); //add the right chunk, 
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: rightChunk
         ensure( leftLeftOver.getLength() > 0, "Invalid lengtgh." );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( leftLeftOver, NEW _Type( *_hint->second ) ) ); //add the leftover
      }

      void insertTotalBeginOverlap()
      {
         MemoryChunk &leftChunk = const_cast< MemoryChunk & >( _hint->first );
         MemoryChunk &rightChunk = _key;
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
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: leftChunk
         _key = rightChunk; // redundant
      }

      void insertSubchunkBeginOverlap()
      {
         MemoryChunk &leftChunk = const_cast< MemoryChunk & >( _hint->first );
         MemoryChunk &rightChunk = _key;
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
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: leftChunk
         ensure( rightChunk.getLength() > 0, "Invalid lengtgh." );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightChunk, NEW _Type( *_hint->second ) ) ); //add the rightChunk
      }

      void insertTotalEndOverlap()
      {
         _Type *ptr = (_Type *) NULL;
         MemoryChunk &leftChunk = _key;
         MemoryChunk &rightChunk = const_cast< MemoryChunk & >( _hint->first );
         const MemoryChunk *rightChunkPtr = &( _hint->first );
         _Type ** rightChunkDataPtr = &( _hint->second );
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
         ensure( leftChunk.getLength() > 0, "Invalid lengtgh." );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( leftChunk, ptr ) ); //add the leftChunk
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: leftChunk
         _ptrList.push_back( MemChunkPair ( rightChunkPtr, rightChunkDataPtr ) ); //return: rightChunk
      }

      void insertSubchunkEndOverlap()
      {
         MemoryChunk &leftChunk = const_cast< MemoryChunk & >( _hint->first );
         MemoryChunk &rightChunk = _key;
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
         ensure( rightChunk.getLength() > 0, "Invalid lengtgh." );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightChunk, NEW _Type( *_hint->second ) ) ); //add the rightChunk
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: rightChunk
      }
   };
   // precondition: there is no exact match
   bool lastChunk = false;
   MemoryChunk iterKey = key;
   LocalFunctions local( *this, iterKey, hint, ptrList );
   if ( !this->empty() )
   {
      if ( hint != this->begin() )
         hint--;
      do {
         if ( hint->first.equal( iterKey ) ) {
            lastChunk = true;
            ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) );
         } else {
            switch ( hint->first.checkOverlap( iterKey ) )
            {
               case MemoryChunk::NO_OVERLAP:
                  if ( iterKey.getAddress() < hint->first.getAddress() )
                  {
                     local.insertNoOverlap();
                     lastChunk = true;
                  }
                  break;
               case MemoryChunk::BEGIN_OVERLAP:
                  local.insertBeginOverlap();
                  lastChunk = true; //this is a final situation since the to-be-inserted chunk ends.
                  break;
               case MemoryChunk::END_OVERLAP:
                  local.insertEndOverlap();
                  break;
               case MemoryChunk::TOTAL_OVERLAP:
                  local.insertTotalOverlap();
                  break;
               case MemoryChunk::SUBCHUNK_OVERLAP:
                  local.insertSubchunkOverlap();
                  lastChunk = true;
                  break;
               case MemoryChunk::TOTAL_BEGIN_OVERLAP:
                  local.insertTotalBeginOverlap();
                  break;
               case MemoryChunk::SUBCHUNK_BEGIN_OVERLAP:
                  local.insertSubchunkBeginOverlap();
                  lastChunk = true;
                  break;
               case MemoryChunk::TOTAL_END_OVERLAP:
                  local.insertTotalEndOverlap();
                  lastChunk = true;
                  break;
               case MemoryChunk::SUBCHUNK_END_OVERLAP:
                  local.insertSubchunkEndOverlap();
                  lastChunk = true;
                  break;
            }
            hint++;
         }
      } while ( hint != this->end() && !lastChunk );
      if ( hint == this->end() && !lastChunk )
      {
         // no more chunks in the map but there is still "key" to insert
         ensure( iterKey.getLength() > 0, "Invalid lengtgh." );
         std::pair<typename BaseMap::iterator, bool> result = this->insert( typename BaseMap::value_type( iterKey, ( _Type * ) NULL ) );
         if ( result.second == false ) {
            *myThread->_file << "ERROR inserting in MemoryMap, key is: " << (void*) iterKey.getAddress() << " : " << iterKey.getLength() << std::endl;
            this->print(*myThread->_file);
         }
         ptrList.push_back( MemChunkPair ( &(result.first->first), &(result.first->second) ) );
      }
   } else {
      ensure( key.getLength() > 0, "Invalid lengtgh." );
      hint = this->insert( hint, typename BaseMap::value_type( key, ( _Type * ) NULL ) );
      ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) );
   }
}


template < typename _Type >
void MemoryMap< _Type >::insertWithOverlapButNotGenerateIntersects( const MemoryChunk &key, typename BaseMap::iterator &hint, MemChunkList &ptrList )
{
   class LocalFunctions {
      MemoryMap< _Type > &_thisMap;
      MemoryChunk        &_key;
      iterator           &_hint;
      MemChunkList       &_ptrList;

      public:
      LocalFunctions( MemoryMap< _Type > &thisMap, MemoryChunk &localKey, iterator &localHint, MemChunkList &localPtrList ) :
         _thisMap( thisMap ), _key( localKey ), _hint( localHint ), _ptrList( localPtrList ) { }

      void insertNoOverlap()
      {
         _Type *ptr = (_Type *) NULL;
         ensure( _key.getLength() > 0, "Invalid lengtgh." );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( _key, ptr ) );
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) );
      }

      void insertBeginOverlap()
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
         MemoryChunk &leftChunk = _key;
         MemoryChunk rightChunk = _hint->first ;
         MemoryChunk rightLeftOver;

         MemoryChunk::intersect( leftChunk, rightChunk, rightLeftOver );

         ensure( leftChunk.getLength() > 0, "Invalid lengtgh." );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( leftChunk, ptr ) ); //add the left chunk, 
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) );
         _hint++; //advance the iterator (it should reach "rightChunk"!)
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) );

         //_hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightLeftOver, NEW _Type( *_hint->second ) ) );
      }

      void insertEndOverlap()
      {
         MemoryChunk leftChunk = _hint->first;
         MemoryChunk &rightChunk = _key;
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
         //_hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightChunk, NEW _Type( *_hint->second ) ) );
         //result: right chunk
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second) ) );
         //leftover not added to result since more overlapping may exist
         _key = rightLeftOver;  //try to add what is left
      }

      void insertTotalOverlap()
      {
         _Type *ptr = (_Type *) NULL;
         MemoryChunk &leftChunk = _key;
         MemoryChunk &rightChunk = const_cast< MemoryChunk & >( _hint->first );
         _Type ** rightChunkDataPtr = &( _hint->second );
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

         ensure( leftChunk.getLength() > 0, "Invalid lengtgh." );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( leftChunk, ptr ) ); //add the left chunk, 
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: leftChunk
         _ptrList.push_back( MemChunkPair ( &rightChunk, rightChunkDataPtr ) ); //return: rightChunk
         _key = leftLeftOver; //try to add what is left
      }

      void insertSubchunkOverlap()
      {
         //MemoryChunk leftChunk = _hint->first;
         //MemoryChunk &rightChunk = _key;
         //MemoryChunk leftLeftOver;
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

         //MemoryChunk::partition( leftChunk, rightChunk, leftLeftOver );

         //_hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightChunk, NEW _Type( *_hint->second ) ) ); //add the right chunk, 
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: rightChunk
         //_hint = _thisMap.insert( _hint, typename BaseMap::value_type( leftLeftOver, NEW _Type( *_hint->second ) ) ); //add the leftover
      }

      void insertTotalBeginOverlap()
      {
         MemoryChunk &leftChunk = const_cast< MemoryChunk & >( _hint->first );
         MemoryChunk &rightChunk = _key;
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
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: leftChunk
         _key = rightChunk; // redundant
      }

      void insertSubchunkBeginOverlap()
      {
         //MemoryChunk &leftChunk = const_cast< MemoryChunk & >( _hint->first );
         //MemoryChunk &rightChunk = _key;
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
         //MemoryChunk::partitionBeginAgtB( leftChunk, rightChunk );
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: leftChunk
         //_hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightChunk, NEW _Type( *_hint->second ) ) ); //add the rightChunk
      }

      void insertTotalEndOverlap()
      {
         _Type *ptr = (_Type *) NULL;
         MemoryChunk &leftChunk = _key;
         MemoryChunk &rightChunk = const_cast< MemoryChunk & >( _hint->first );
         const MemoryChunk *rightChunkPtr = &( _hint->first );
         _Type ** rightChunkDataPtr = &( _hint->second );
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
         ensure( leftChunk.getLength() > 0, "Invalid lengtgh." );
         _hint = _thisMap.insert( _hint, typename BaseMap::value_type( leftChunk, ptr ) ); //add the leftChunk
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: leftChunk
         _ptrList.push_back( MemChunkPair ( rightChunkPtr, rightChunkDataPtr ) ); //return: rightChunk
      }

      void insertSubchunkEndOverlap()
      {
         //MemoryChunk leftChunk = _hint->first;
         //MemoryChunk &rightChunk = _key;
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
         //MemoryChunk::partitionEnd( leftChunk, rightChunk );
         //_hint = _thisMap.insert( _hint, typename BaseMap::value_type( rightChunk, NEW _Type( *_hint->second ) ) ); //add the rightChunk
         _ptrList.push_back( MemChunkPair ( &( _hint->first ), &( _hint->second ) ) ); //return: rightChunk
      }
   };
   // precondition: there is no exact match
   bool lastChunk = false;
   MemoryChunk iterKey = key;
   LocalFunctions local( *this, iterKey, hint, ptrList );
   if ( !this->empty() )
   {
      if ( hint != this->begin() )
         hint--;
      do {
         if ( hint->first.equal( iterKey ) ) {
            lastChunk = true;
            ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) );
         } else {
            switch ( hint->first.checkOverlap( iterKey ) )
            {
               case MemoryChunk::NO_OVERLAP:
                  if ( iterKey.getAddress() < hint->first.getAddress() )
                  {
                     local.insertNoOverlap();
                     lastChunk = true;
                  }
                  break;
               case MemoryChunk::BEGIN_OVERLAP:
                  local.insertBeginOverlap();
                  lastChunk = true; //this is a final situation since the to-be-inserted chunk ends.
                  break;
               case MemoryChunk::END_OVERLAP:
                  local.insertEndOverlap();
                  break;
               case MemoryChunk::TOTAL_OVERLAP:
                  local.insertTotalOverlap();
                  break;
               case MemoryChunk::SUBCHUNK_OVERLAP:
                  local.insertSubchunkOverlap();
                  lastChunk = true;
                  break;
               case MemoryChunk::TOTAL_BEGIN_OVERLAP:
                  local.insertTotalBeginOverlap();
                  break;
               case MemoryChunk::SUBCHUNK_BEGIN_OVERLAP:
                  local.insertSubchunkBeginOverlap();
                  lastChunk = true;
                  break;
               case MemoryChunk::TOTAL_END_OVERLAP:
                  local.insertTotalEndOverlap();
                  lastChunk = true;
                  break;
               case MemoryChunk::SUBCHUNK_END_OVERLAP:
                  local.insertSubchunkEndOverlap();
                  lastChunk = true;
                  break;
            }
            hint++;
         }
      } while ( hint != this->end() && !lastChunk );
      if ( hint == this->end() && !lastChunk )
      {
         // no more chunks in the map but there is still "key" to insert
         ensure( iterKey.getLength() > 0, "Invalid lengtgh." );
         hint = this->insert( hint, typename BaseMap::value_type( iterKey, ( _Type * ) NULL ) );
         ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) );
      }
   } else {
      ensure( key.getLength() > 0, "Invalid lengtgh." );
      hint = this->insert( hint, typename BaseMap::value_type( key, ( _Type * ) NULL ) );
      ptrList.push_back( MemChunkPair ( &(hint->first), &(hint->second) ) );
   }
}

template < typename _Type >
void MemoryMap< _Type >::getWithOverlapNoExactKey( const MemoryChunk &key, const_iterator &hint, ConstMemChunkList &ptrList ) const
{
   class LocalFunctions {
      MemoryChunk          &_key;
      const_iterator       &_hint;
      ConstMemChunkList    &_ptrList;

      public:
      LocalFunctions( MemoryChunk &localKey, const_iterator &localHint, ConstMemChunkList &localPtrList ) :
         _key( localKey ), _hint( localHint ), _ptrList( localPtrList ) { }

      void getNoOverlap()
      {
         _ptrList.push_back( ConstMemChunkPair ( MemoryChunk( _key ), NULL ) );
      }

      void getBeginOverlap( )
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
         MemoryChunk leftChunk = _key;
         MemoryChunk rightChunk = _hint->first ;
         MemoryChunk rightLeftOver;

         MemoryChunk::intersect( leftChunk, rightChunk, rightLeftOver );

         _ptrList.push_back( ConstMemChunkPair ( MemoryChunk( leftChunk ), NULL ) );
         _ptrList.push_back( ConstMemChunkPair ( MemoryChunk( _hint->first ), _hint->second ) );
      }

      void getEndOverlap( )
      {
         MemoryChunk leftChunk = _hint->first ;
         MemoryChunk rightChunk = _key;
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
         _ptrList.push_back( ConstMemChunkPair ( MemoryChunk( _hint->first ), _hint->second) );
         //leftover not added to result since more overlapping may exist
         _key = rightLeftOver;  //try to add what is left
      }

      void getTotalOverlap( )
      {
         //_Type *ptr = (_Type *) NULL;
         MemoryChunk leftChunk = _key;
         MemoryChunk rightChunk = _hint->first ;
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

         _ptrList.push_back( ConstMemChunkPair ( MemoryChunk ( leftChunk ), NULL ) ); //return: leftChunk
         _ptrList.push_back( ConstMemChunkPair ( MemoryChunk ( _hint->first ), _hint->second ) ); //return: rightChunk
         _key = leftLeftOver; //try to add what is left
      }

      void getSubchunkOverlap()
      {
         //MemoryChunk leftChunk = _hint->first ;
         //MemoryChunk rightChunk = _key;
         //MemoryChunk leftLeftOver;
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

         //MemoryChunk::partition( leftChunk, rightChunk, leftLeftOver );

         _ptrList.push_back( ConstMemChunkPair ( MemoryChunk( _hint->first ), _hint->second ) ); //return: rightChunk
      }

      void getTotalBeginOverlap()
      {
         MemoryChunk leftChunk = _hint->first ;
         MemoryChunk rightChunk = _key;
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
         _ptrList.push_back( ConstMemChunkPair ( MemoryChunk ( _hint->first ), _hint->second ) ); //return: leftChunk
         _key = rightChunk;
      }

      void getSubchunkBeginOverlap()
      {
         //MemoryChunk leftChunk = _hint->first ;
         //MemoryChunk rightChunk = _key;
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
         //MemoryChunk::partitionBeginAgtB( leftChunk, rightChunk );
         _ptrList.push_back( ConstMemChunkPair ( MemoryChunk( _hint->first ), _hint->second ) ); //return: leftChunk
      }

      void getTotalEndOverlap()
      {
         MemoryChunk leftChunk = _key;
         MemoryChunk rightChunk = _hint->first ;
         //const MemoryChunk *rightChunkPtr = &( hint->first );
         _Type * rightChunkData = _hint->second;
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
         _ptrList.push_back( ConstMemChunkPair ( MemoryChunk ( leftChunk ), NULL ) ); //return: leftChunk
         _ptrList.push_back( ConstMemChunkPair ( MemoryChunk ( _hint->first ), rightChunkData ) ); //return: rightChunk
      }

      void getSubchunkEndOverlap()
      {
         //MemoryChunk leftChunk = _hint->first ;
         //MemoryChunk rightChunk = _key;
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
         //MemoryChunk::partitionEnd( leftChunk, rightChunk );
         _ptrList.push_back( ConstMemChunkPair ( MemoryChunk ( _hint->first ), _hint->second ) ); //return: rightChunk
      }
   };

   // precondition: there is no exact match
   bool lastChunk = false;
   MemoryChunk iterKey = key;
   LocalFunctions local( iterKey, hint, ptrList );
   if ( !this->empty() )
   {
      if ( hint != this->begin() ) {
         hint--;
      }
      do {
         //*myThread->_file << "iterKey " << iterKey.getAddress() << " " << iterKey.getLength() << ", hint " << hint->first.getAddress() << " " << hint->first.getLength() << " " << MemoryChunk::strOverlap[ hint->first.checkOverlap( iterKey ) ] << std::endl;
         if ( hint->first.equal( iterKey ) ) {
            lastChunk = true;
            ptrList.push_back( ConstMemChunkPair ( MemoryChunk( hint->first ), hint->second ) );
         } else {
            switch ( hint->first.checkOverlap( iterKey ) )
            {
               case MemoryChunk::NO_OVERLAP:
                  if ( iterKey.getAddress() < hint->first.getAddress() )
                  {
                     local.getNoOverlap();
                     lastChunk = true;
                  }
                  break;
               case MemoryChunk::BEGIN_OVERLAP:
                  local.getBeginOverlap();
                  lastChunk = true; //this is a final situation since the to-be-inserted chunk ends.
                  break;
               case MemoryChunk::END_OVERLAP:
                  local.getEndOverlap();
                  break;
               case MemoryChunk::TOTAL_OVERLAP:
                  local.getTotalOverlap();
                  break;
               case MemoryChunk::SUBCHUNK_OVERLAP:
                  local.getSubchunkOverlap();
                  lastChunk = true;
                  break;
               case MemoryChunk::TOTAL_BEGIN_OVERLAP:
                  local.getTotalBeginOverlap();
                  break;
               case MemoryChunk::SUBCHUNK_BEGIN_OVERLAP:
                  local.getSubchunkBeginOverlap();
                  lastChunk = true;
                  break;
               case MemoryChunk::TOTAL_END_OVERLAP:
                  local.getTotalEndOverlap();
                  lastChunk = true;
                  break;
               case MemoryChunk::SUBCHUNK_END_OVERLAP:
                  local.getSubchunkEndOverlap();
                  lastChunk = true;
                  break;
            }
            hint++;
         }
      } while ( hint != this->end() && !lastChunk );
   } else {
      ptrList.push_back( ConstMemChunkPair ( MemoryChunk(key), NULL ) );
   }
}

template < typename _Type >
void MemoryMap< _Type >::getOrAddChunkDoNotFragment( uint64_t addr, std::size_t len, MemChunkList &resultEntries )
{
   MemoryChunk key( addr, len );

   typename BaseMap::iterator it = this->lower_bound( key );
   if ( it == this->end() || this->key_comp()( key, it->first ) || it->first.getLength() != len )
   {
      /* NOT EXACT ADDR FOUND: "addr" is higher than any odther addr in the map OR less than "it" */
      insertWithOverlapButNotGenerateIntersects( key, it, resultEntries );
      //res = MEM_CHUNK_NOT_FOUND_BUT_ALLOCATED;
   }
   else
   {
      /* EXACT ADDR FOUND */
      resultEntries.push_back( MemChunkPair( &(it->first), &(it->second) ) );
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
void MemoryMap< _Type >::getChunk( uint64_t addr, std::size_t len, ConstMemChunkList &resultEntries ) const
{
   MemoryChunk key( addr, len );

   //*myThread->_file << __func__ << " key requested addr " << key.getAddress() << ", len "  << key.getLength() << std::endl;
   const_iterator it = this->lower_bound( key );
   if ( it == this->end() || this->key_comp()( key, it->first ) || it->first.getLength() != len )
   {
      /* NOT EXACT ADDR FOUND: "addr" is higher than any odther addr in the map OR less than "it" */
      getWithOverlapNoExactKey( key, it, resultEntries );
      //for (typename ConstMemChunkList::const_iterator __it = resultEntries.begin(); __it != resultEntries.end(); __it++)
      //{
      //   *myThread->_file << "result entry addr " << __it->first->getAddress() << ", len " << __it->first->getLength();
      //   //fprintf(stderr, "result entry addr %zu, len %zu\n", __it->first->getAddress(), __it->first->getLength() );
      //   if ( __it->second != NULL ) 
      //      *myThread->_file << "[ AllocChunk@: " << (*__it->second) << " ]" << std::endl;// fprintf(stderr, "AllocChunk@: %p\n", (*__it->second));
      //   else *myThread->_file << "[ null ]" << std::endl; //fprintf(stderr, "entry ptr is null!! \n");
      //}
      //if ( resultEntries.size() == 0 )
      //   *myThread->_file << "result entry EMPTY!" << std::endl;//fprintf(stderr, "result entry EMPTY!\n" );

   }
   else
   {
      /* EXACT ADDR FOUND */
      resultEntries.push_back( ConstMemChunkPair( MemoryChunk( key ), it->second ) );
      //fprintf(stderr, "result entry addr %d, len %d\n", key.getAddress(), key.getLength() );
   }
}

template < typename _Type >
void MemoryMap< _Type >::print(std::ostream &o) const
{
   int i = 0;
   typename BaseMap::const_iterator it = this->begin();
   o << "printing memory chunks" << std::endl;
   for (; it != this->end(); it++)
   {
      o << "\tchunk: " << i++ << " addr=" << (void *) it->first.getAddress() <<"(" << (uint64_t)it->first.getAddress()<< ")" << " len=" << it->first.getLength() << " ptr val is " << it->second << " addr of ptr val is " << (void *) &(it->second) << " ";
      o << std::endl;
      //it->second->print();
   } 
   o << "end of memory chunks" << std::endl;
}

template < typename _Type >
bool MemoryMap< _Type >::canPack() const
{
   std::list< std::pair< MemoryChunk *, std::size_t > > accesses;
   bool result = true;
   if ( this->empty() ) {
      std::cerr <<"cant pack, empty" << std::endl;
      result = false;
   } else if ( this->size() == 1 ) {
      std::cerr <<"cant pack, size = 1" << std::endl;
      result = false;
   } else {

   //{
   //for (const_iterator itp = this->begin(); itp != this->end(); itp++ ) { std::cerr << "addr=" << (void*)itp->first.getAddress() << " len="<<itp->first.getLength() << " "; }
   //std::cerr << "EOL"<< std::endl;
   //}

   //{
   //   const_iterator it = this->begin();;

   //   MemoryChunk *mc = &(it->first);
   //   std::size_t iterSize, currSize, count = 1;
   //   uint64_t iterAddr, currAddr;
   //      currSize = it->first.getLength();
   //      currAddr = it->first.getAddress();
   //      it++;
   //      iterSize = it->first.getLength();
   //      iterAddr = it->first.getAddress();
   //   do {
   //      
   //      if ( iterSize != currSize &&  ) { accesses.push_back( std::make_pair( mc, count ) ); currSize = iterSize; currAddr = iterAddr; } 
   //      else { count += 1; }
   //   } while( it != this->end() );
   //}

#if 0
   const_iterator it = this->begin();
   uint64_t firstAddr = it->first.getAddress();
   std::size_t firstSize = it->first.getLength();

   it++;
   
   uint64_t diffAddr = it->first.getAddress() - firstAddr;
   std::size_t diffSize = it->first.getLength() - firstSize;

   std::cerr << "First diff addr is "<< diffAddr << std::endl;
   if ( diffSize != 0 ) result = false;

   uint64_t iterAddr = it->first.getAddress();
   std::size_t iterSize = it->first.getLength();
   it++;

   for (; it != this->end() && result == true ; it++ ) {
      result = result && ( diffSize == ( it->first.getLength() - iterSize ) );
      result = result && ( diffAddr == ( it->first.getAddress() - iterAddr ) );
      if ( result ) {
      iterAddr = it->first.getAddress();
      iterSize = it->first.getLength();
      } else {std::cerr <<"cant pack, diff size " << diffSize << " and iter diff " << ( it->first.getLength() - iterSize )  <<" diff Addr " <<diffAddr << " and iterdiff "<< ( it->first.getAddress() - iterAddr )   << std::endl;
      }
   }

   if ( result ) std::cerr << " Seems I can pack, diffAddr is " << diffAddr << " and diffSize is " << diffSize << std::endl;
   else { it--; std::cerr <<"cant pack, diff size " << diffSize << " and iter diff " << ( it->first.getLength() - iterSize )  <<" diff Addr " <<diffAddr << " and iterdiff "<< ( it->first.getAddress() - iterAddr )   << std::endl; }
   r
#endif

   }
   return result;
}


template < typename _Type >
void MemoryMap< _Type >::removeChunks( uint64_t addr, std::size_t len ) {
   MemoryChunk key( addr, len );
   typename BaseMap::iterator it_begin, it_end;
   it_begin = it_end = this->lower_bound( key );
   while ( it_end != this->end() && key.contains( it_end->first ) ) {
      it_end++;
   }
   this->erase( it_begin, it_end );
}

template < typename _Type >
_Type **MemoryMap< _Type >::getExactOrFullyOverlappingInsertIfNotFound( uint64_t addr, std::size_t len, bool &exact ) {
   _Type **ptr = (_Type **) NULL;
   MemoryChunk key( addr, len );
   iterator it = this->lower_bound( key );
   if ( it == this->end() )
   {
      /* not exact match found, check it-- */
      if ( this->size() == 0 ) {
         ensure( key.getLength() > 0, "Invalid lengtgh." );
         it = this->insert( it, typename BaseMap::value_type( key, ( _Type * ) NULL ) );
         ptr = &( it->second );
         exact = true;
      } else {
         it--;
         MemoryChunk::OverlapType ov = it->first.checkOverlap( key );
         if ( ov == MemoryChunk::NO_OVERLAP ) {
            ensure( key.getLength() > 0, "Invalid lengtgh." );
            it = this->insert( it, typename BaseMap::value_type( key, ( _Type * ) NULL ) );
            ptr = &( it->second );
            exact = true;
         } else if ( ov == MemoryChunk::SUBCHUNK_OVERLAP ||
               ov == MemoryChunk::SUBCHUNK_BEGIN_OVERLAP ||
               ov == MemoryChunk::SUBCHUNK_END_OVERLAP) {
            ptr = &( it->second );
            exact = false;
         }
      }
   } else if ( this->key_comp()( key, it->first ) ) {
      /* key less than it->first, not total overlap guaranteed */
      if ( it->first.checkOverlap( key ) == MemoryChunk::NO_OVERLAP ) {
         if ( it == this->begin() ) {
            ensure( key.getLength() > 0, "Invalid lengtgh." );
            it = this->insert( it, typename BaseMap::value_type( key, ( _Type * ) NULL ) );
            ptr = &( it->second );
            exact = true;
         } else {
            it--;
            MemoryChunk::OverlapType ov = it->first.checkOverlap( key );
            if ( ov == MemoryChunk::NO_OVERLAP ) {
               ensure( key.getLength() > 0, "Invalid lengtgh." );
               it = this->insert( it, typename BaseMap::value_type( key, ( _Type * ) NULL ) );
               ptr = &( it->second );
               exact = true;
            } else if ( ov == MemoryChunk::SUBCHUNK_OVERLAP ||
                  ov == MemoryChunk::SUBCHUNK_BEGIN_OVERLAP ||
                  ov == MemoryChunk::SUBCHUNK_END_OVERLAP) {
               ptr = &( it->second );
               exact = false;
            }
            //it = this->insert( it, typename BaseMap::value_type( key, ( _Type * ) NULL ) );
            //ptr = &( it->second );
            //exact = true;
         }
      } else {
         ptr = NULL;
         exact = false;
      }
   } else if ( it->first.getLength() != len ) {
      /* same addr, wrong length */
      MemoryChunk::OverlapType ov = it->first.checkOverlap( key );
      if ( ov == MemoryChunk::SUBCHUNK_OVERLAP ||
            ov == MemoryChunk::SUBCHUNK_BEGIN_OVERLAP ||
            ov == MemoryChunk::SUBCHUNK_END_OVERLAP) {
         ptr = &( it->second );
         exact = false;
      } else {
         ptr = NULL;
         exact = false;
      }
   } else {
      /* exact address found */
      ptr = &( it->second );
      exact = true;
   }
   return ptr;
}

template < typename _Type >
_Type **MemoryMap< _Type >::getExactInsertIfNotFound( uint64_t addr, std::size_t len ) {
   _Type **ptr = (_Type **) NULL;
   MemoryChunk key( addr, len );
   iterator it = this->lower_bound( key );
   if ( it == this->end() )
   {
      /* not exact match found, check it-- */
      if ( this->size() == 0 ) {
         ensure( key.getLength() > 0, "Invalid lengtgh." );
         it = this->insert( it, typename BaseMap::value_type( key, ( _Type * ) NULL ) );
         ptr = &( it->second );
      } else {
         it--;
         if ( it->first.checkOverlap( key ) == MemoryChunk::NO_OVERLAP ) {
            ensure( key.getLength() > 0, "Invalid lengtgh." );
            it = this->insert( it, typename BaseMap::value_type( key, ( _Type * ) NULL ) );
            ptr = &( it->second );
         } else {
            ptr = NULL;
         }
      }
   } else if ( this->key_comp()( key, it->first ) ) {
      /* less than addr */
      if ( it->first.checkOverlap( key ) == MemoryChunk::NO_OVERLAP ) {
         ensure( key.getLength() > 0, "Invalid lengtgh." );
         it = this->insert( it, typename BaseMap::value_type( key, ( _Type * ) NULL ) );
         ptr = &( it->second );
      } else {
         ptr = NULL;
      }
   } else if ( it->first.getLength() != len ) {
      /* same addr, wrong length */
      ptr = NULL;
   } else {
      /* exact address found */
      ptr = &( it->second );
   }
   return ptr;
}

template < typename _Type >
_Type *MemoryMap< _Type >::getExactByAddress( uint64_t addr ) const {
   _Type *ptr = (_Type *) NULL;
   MemoryChunk key( addr, 0 );
   const_iterator it = this->lower_bound( key );
   if ( it == this->end() || this->key_comp()( key, it->first ) )
   {
      /* not exact match found */
      ptr = NULL;
   } else {
      /* exact address found */
      ptr = it->second;
   }
   return ptr;
}

template < typename _Type >
void MemoryMap< _Type >::eraseByAddress( uint64_t addr ) {
   MemoryChunk key( addr, 0 );
   iterator it = this->lower_bound( key );
   if ( it == this->end() || this->key_comp()( key, it->first ) )
   {
      std::cerr << "Could not erase, address not found." << std::endl;
      exit(-1);
   } else {
      this->erase( it );
   }
}

} // namespace nanos

#endif /* _NANOS_MEMORYMAP_H */
