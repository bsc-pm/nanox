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

#ifndef _NANOS_MERGEABLE_MEMORYMAP_H
#define _NANOS_MERGEABLE_MEMORYMAP_H

#include "memorymap.hpp"
#include "mergeablememorymap_decl.hpp"

namespace nanos {

/* merge input MemoryMap mm into this */
template < typename _Type >
void MergeableMemoryMap< _Type >::merge( const MemoryMap< _Type > &mm )
{
   const_iterator inputIt = mm.begin();

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

/* merge input MemoryMap mm into this */
template < typename _Type >
void MergeableMemoryMap< _Type >::merge2( const MemoryMap< _Type > &mm )
{
   class LocalFunctions {
      MemoryMap<_Type> &_thisMap;

      public:
      LocalFunctions( MemoryMap<_Type> &thisMap ) : _thisMap( thisMap ) { }

      void tryToMergeWithPreviousEntry( iterator &thisIt ) {
         if ( thisIt != _thisMap.begin() ) {
            iterator prevIt = thisIt;
            prevIt--;
            if ( prevIt->first.getAddress() + prevIt->first.getLength() == thisIt->first.getAddress() ) {
               if ( prevIt->second->equal( *thisIt->second ) ) { 
                  MemoryChunk &prevNoConst = const_cast< MemoryChunk & >( prevIt->first );
                  prevNoConst.expandIncluding( thisIt->first );
                  _thisMap.erase( thisIt );
                  thisIt = _thisMap.find( prevNoConst );
               }
            }
         }
      }

      void expand( MemoryChunk &inputKey, _Type * const &inputData, iterator &thisIt )
      {
         class ExpandLocalFunctions {

            MemoryMap< _Type > &_thisMap;
            MemoryChunk        &_inputKey;
            _Type *      const &_inputData;
            iterator           &_thisIt;

            bool _thisAndInputDataAreEqual;
            bool _nextAndInputDataAreEqual;

            public:
            ExpandLocalFunctions( MemoryMap<_Type> &thisMap, MemoryChunk &localInputKey, _Type * const &localInputData, iterator &localThisIt ) :
               _thisMap( thisMap ), _inputKey( localInputKey ), _inputData( localInputData ), _thisIt( localThisIt ) {
                  _thisAndInputDataAreEqual = _thisIt->second->equal( *_inputData );
               }

            void expandNoOverlap() {
               if ( _thisAndInputDataAreEqual ) {
                  /* we can expand safetly, and we finish */
                  MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                  thisNoConst.expandIncluding( _inputKey );
               } else {
                  _thisIt = _thisMap.insert( _thisIt, typename BaseMap::value_type( _inputKey, NEW _Type ( *_inputData ) ) );
               }
               _thisIt++;
            }

            void expandBeginOverlap( iterator &nextIt ) {
               const MemoryChunk &nextKey = nextIt->first;
               const MemoryChunk &thisKey = _thisIt->first;
               _nextAndInputDataAreEqual = nextIt->second->equal( *_inputData );
               if ( _thisAndInputDataAreEqual ) {
                  /* if the chunk Data is equivalent, we can expan all chunks and we finish */
                  if ( _nextAndInputDataAreEqual )
                  {
                     /* expand to fit [this ... inputKey ... next]
                      * then erase next entry
                      */
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                     thisNoConst.expandIncluding( nextKey );
                     delete nextIt->second;
                     _thisMap.erase( nextIt );
                     _thisIt = _thisMap.find( thisKey );
                  } else {
                     /* fit [this ... inputKey ... ]{next:overlap_with_input}(next:leftover)
                      */
                     MemoryChunk rightLeftOver;
                     MemoryChunk inputKeyCopy = _inputKey;
                     MemoryChunk &rightChunk = const_cast< MemoryChunk & >( nextIt->first );
                     /* we can not merge both already-inserted entries,
                      * expand "this" as much as possible and partition
                      * the leftover of "inputKey" that intersects with hintCopy
                      */
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                     thisNoConst.expandExcluding( nextIt->first );
                     MemoryChunk::intersect( inputKeyCopy, rightChunk, rightLeftOver ); //this modifies nextIt->first!!
                     _thisIt = _thisMap.insert( nextIt, typename BaseMap::value_type( rightLeftOver, NEW _Type( *nextIt->second ) ) );
                     _thisIt--;
                     _thisIt->second->merge( *_inputData );
                     _thisIt++;
                  }
               } else {
                  if ( _nextAndInputDataAreEqual )
                  {
                     /* expand to fit: (this)[... inputKey ... next]
                      */
                     std::pair<typename BaseMap::iterator, bool> insertResult;
                     _inputKey.expandIncluding( nextKey );
                     _Type *data = nextIt->second;
                     _thisMap.erase( nextIt );
                     insertResult = _thisMap.insert( typename BaseMap::value_type( _inputKey, data ) ); //reuse data entry
                     _thisIt = insertResult.first;
                  } else {
                     /* fit (this)[ ... inputKey ... ]{next:overlap_with_input}(next:leftover)
                      */
                     MemoryChunk rightLeftOver;
                     MemoryChunk inputKeyCopy = _inputKey;
                     MemoryChunk &rightChunk = const_cast< MemoryChunk & >( nextIt->first );
                     /* we can not merge both already-inserted entries,
                      * expand "this" as much as possible and partition
                      * the leftover of "inputKey" that intersects with hintCopy
                      */
                     MemoryChunk::intersect( inputKeyCopy, rightChunk, rightLeftOver ); //this modifies nextIt->first!!
                     _thisIt = _thisMap.insert( _thisIt, typename BaseMap::value_type( inputKeyCopy, NEW _Type( *_inputData ) ) );
                     _thisIt++;
                     _thisIt = _thisMap.insert( _thisIt, typename BaseMap::value_type( rightLeftOver, NEW _Type( *_thisIt->second ) ) );
                     _thisIt--;
                     _thisIt->second->merge( *_inputData ); //merge overlapped chunk
                     _thisIt++;
                  }
               }
            }

            void expandTotalOverlap( iterator &nextIt ) {
               const MemoryChunk &nextKey = nextIt->first;
               _nextAndInputDataAreEqual = nextIt->second->equal( *_inputData );
               if ( _thisAndInputDataAreEqual ) {
                  MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                  if ( _nextAndInputDataAreEqual )
                  {
                     /* [this .. inputKey .. next]{..inputKey..} */
                     thisNoConst.expandIncluding( nextKey );
                     MemoryChunk key = _thisIt->first;
                     delete nextIt->second;
                     _thisMap.erase( nextIt );
                     _thisIt = nextIt = _thisMap.find( key );
                     //nextIt++;
                     _inputKey.cutAfter( _thisIt->first );
                  } else {
                     /* [this .. inputKey .. ][next]{..inputKey..} */
                     thisNoConst.expandExcluding( nextKey );
                     nextIt->second->merge( *_inputData );
                     _inputKey.cutAfter( nextKey );
                     _thisIt++; //increase thisIt to keep processing
                  }
               } else {
                  if ( _nextAndInputDataAreEqual )
                  {
                     /* (this)[.. inputKey .. next]{..inputKey..} */
                     std::pair<typename BaseMap::iterator, bool> insertResult;
                     MemoryChunk key = _inputKey;
                     key.expandIncluding( nextKey );
                     _Type *data = nextIt->second;
                     _thisMap.erase( nextIt );
                     insertResult = _thisMap.insert( typename BaseMap::value_type( _inputKey, data ) ); //reuse data entry
                     _thisIt = insertResult.first;
                     _inputKey.cutAfter( _thisIt->first );
                  } else {
                     /* (this)[.. inputKey .. ][next]{inputKey} */
                     MemoryChunk key = _inputKey;
                     key.expandExcluding( nextKey );
                     nextIt->second->merge( *_inputData );
                     _inputKey.cutAfter( nextKey );
                     _thisIt = _thisMap.insert( _thisIt, typename BaseMap::value_type( key, NEW _Type( *_inputData ) ) );
                     _thisIt++;
                  }
               }
            }

            void expandTotalBeginOverlap( iterator &nextIt ) {
               const MemoryChunk &nextKey = nextIt->first;
               const MemoryChunk &thisKey = _thisIt->first;
               _nextAndInputDataAreEqual = nextIt->second->equal( *_inputData );
               if ( _thisAndInputDataAreEqual ) {
                  if ( _nextAndInputDataAreEqual ) {
                     /* expand to fit [this][next]{next:leftover}
                      * then erase next entry
                      */
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                     thisNoConst.expandIncluding( nextKey );
                     delete nextIt->second;
                     _thisMap.erase( nextIt );
                     _thisIt = _thisMap.find( thisKey );
                  } else {
                     /* fit (this){next:partition_with_input}(next:leftover)
                      * merge partition, continue processing leftover
                      */
                     nextIt->second->merge( *_inputData );
                  }
               } else {
                  if ( _nextAndInputDataAreEqual ) {
                     /* expand to fit (this)[next]{next:leftover}
                      * since next and input are equal, no action is required
                      */
                  } else {
                     /* expand to fit (this)[next]{next:leftover}
                      * similar than before but we need to merge the information
                      * of "next" and "input".
                      */
                     nextIt->second->merge( *_inputData );
                  }
               }
               _inputKey.cutAfter( nextKey );
               _thisIt++;
            }

            void expandSubchunkBeginOverlap( iterator &nextIt )
            {
               MemoryChunk nextKeyCopy = nextIt->first;
               const MemoryChunk &thisKey = _thisIt->first;
               _Type *nextData = nextIt->second;
               _Type *thisData = _thisIt->second;
               _nextAndInputDataAreEqual = nextIt->second->equal( *_inputData );
               if ( _thisAndInputDataAreEqual ) {
                  if ( _nextAndInputDataAreEqual ) {
                     MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                     thisNoConst.expandIncluding( nextKeyCopy );
                     _thisMap.erase( nextIt );
                  } else {
                     if ( !nextData->contains( *_inputData ) ) {
                        MemoryChunk &nextNoConst = const_cast< MemoryChunk & >( nextIt->first );
                        nextNoConst = _inputKey;
                        nextKeyCopy.cutAfter( _inputKey );
                        _thisIt = _thisMap.insert( nextIt, typename BaseMap::value_type( nextKeyCopy, NEW _Type ( *nextData ) ) );
                        nextData->merge( *_inputData ); //merge after insert to create a copy of the original "next" entry.
                     } //else No action is needed because merge would not create a new entry
                  }
               } else {
                  if ( _nextAndInputDataAreEqual ) {
                     //no expand with "this" and "next" already ok, do nothing
                  } else {
                     _Type * tmpData = NEW _Type ( *_inputData );
                     tmpData->merge( *nextData );
                     if ( tmpData->equal( *thisData ) ) {
                        if ( thisData->equal( *nextData ) ) { //merge Equal with BOTH next and this
                           MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                           thisNoConst.expandIncluding( nextKeyCopy );
                           _thisMap.erase( nextIt );
                           _thisIt = _thisMap.find( thisKey );
                        } else {
                           std::pair<typename BaseMap::iterator, bool> insertResult;
                           MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                           thisNoConst.expandIncluding( _inputKey );
                           _thisMap.erase( nextIt );
                           MemoryChunk::partitionBeginAgtB( nextKeyCopy, _inputKey );
                           insertResult = _thisMap.insert( typename BaseMap::value_type( _inputKey, nextData ) ); //reuse entry
                           _thisIt = insertResult.first;
                           delete tmpData;
                        }
                     } else {
                        MemoryChunk &nextNoConst = const_cast< MemoryChunk & >( nextIt->first );
                        nextNoConst = _inputKey;
                        nextKeyCopy.cutAfter( _inputKey );
                        _thisIt = _thisMap.insert( nextIt, typename BaseMap::value_type( nextKeyCopy, tmpData ) );
                     }
                  }
               }
            }

            void expandTotalEndOverlap( iterator &nextIt ) {
               const MemoryChunk &nextKey = nextIt->first;
               _nextAndInputDataAreEqual = nextIt->second->equal( *_inputData );
               if ( _thisAndInputDataAreEqual ) {
                  MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( _thisIt->first );
                  if ( _nextAndInputDataAreEqual ) {
                     thisNoConst.expandIncluding( nextKey );
                     delete nextIt->second;
                     _thisMap.erase( nextIt );
                     _thisIt = _thisMap.find( thisNoConst );
                  } else {
                     thisNoConst.expandExcluding( nextKey );
                     nextIt->second->merge( *_inputData );
                  }
               } else {
                  if ( _nextAndInputDataAreEqual ) {
                     _thisIt = _thisMap.insert( _thisIt, typename BaseMap::value_type( _inputKey, nextIt->second ) );
                     _thisIt++;
                     _thisMap.erase( _thisIt );
                     _thisIt = _thisMap.find( _inputKey );
                  } else {
                     _inputKey.expandExcluding( nextKey );
                     nextIt->second->merge( *_inputData );
                     _thisIt = _thisMap.insert( _thisIt, typename BaseMap::value_type( _inputKey, NEW _Type ( *_inputData ) ) );
                  }
               }
               _thisIt++;
            }
         };

         ExpandLocalFunctions localexp ( _thisMap, inputKey, inputData, thisIt );

         iterator nextIt = thisIt;
         nextIt++;
         _Type *thisData = thisIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         //fprintf(stderr, "entry %s, this %d-%d  input %d-%d\n", __FUNCTION__, thisKey.getAddress(), thisKey.getLength(), inputKey.getAddress(), inputKey.getLength() );

         if ( nextIt == _thisMap.end() )
         {
            if ( thisAndInputDataAreEqual ) {
               /* we can expand safetly, and we finish */
               MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
               thisNoConst.expandIncluding( inputKey );
            } else {
               thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type ( *inputData ) ) );
            }
            thisIt++;
         } else {
            MemoryChunk nextKey = nextIt->first;
            _Type *nextData = nextIt->second;
            bool nextAndInputDataAreEqual = nextData->equal( *inputData );

            if ( nextKey.equal( inputKey ) ) {
               //fprintf(stderr, "EQ! entry %s, next %d-%d  input %d-%d\n", __FUNCTION__, nextKey.getAddress(), nextKey.getLength(), inputKey.getAddress(), inputKey.getLength() );
               if ( thisAndInputDataAreEqual && nextAndInputDataAreEqual ) 
                  fprintf(stderr, "ERROR: I think this is an error. input, and next are equal! data also (this, next and input)\n");
               if ( !nextAndInputDataAreEqual ) {
                  nextData->merge( *inputData );
               }
               tryToMergeWithPreviousEntry( nextIt );
               nextIt++;
               if ( nextIt != _thisMap.end() ) {
                  tryToMergeWithPreviousEntry( nextIt );
               }
            } else {
               fprintf(stderr, "entry %s, next %ld-%ld  input %ld-%ld, overlap is %s\n", __FUNCTION__, nextKey.getAddress(), nextKey.getLength(), inputKey.getAddress(), inputKey.getLength(), MemoryChunk::strOverlap[ nextKey.checkOverlap( inputKey ) ] );
               /* check overlap with the next item in the map! */
               switch ( nextKey.checkOverlap( inputKey ) ) {
                  case MemoryChunk::NO_OVERLAP:
                     localexp.expandNoOverlap();
                     break;
                  case MemoryChunk::BEGIN_OVERLAP:
                     localexp.expandBeginOverlap( nextIt );
                     break;
                  case MemoryChunk::END_OVERLAP:
                     /* ERROR, it is impossible to have End Overlap with hintNext */
                     fprintf(stderr, "error END_OVERLAP %s\n", __FUNCTION__ );
                     break;
                  case MemoryChunk::TOTAL_OVERLAP:
                     localexp.expandTotalOverlap( nextIt );
                     expand( inputKey, inputData, thisIt );
                     break;
                  case MemoryChunk::SUBCHUNK_OVERLAP:
                     /* ERROR, it is impossible to have Subchunk Overlap with hintNext */
                     fprintf(stderr, "error SUBCHUNK_OVERLAP %s\n", __FUNCTION__ );
                     break;
                  case MemoryChunk::TOTAL_BEGIN_OVERLAP:
                     localexp.expandTotalBeginOverlap( nextIt );
                     expand( inputKey, inputData, nextIt );
                     break;
                  case MemoryChunk::SUBCHUNK_BEGIN_OVERLAP:
                     localexp.expandSubchunkBeginOverlap( nextIt );
                     break;
                  case MemoryChunk::TOTAL_END_OVERLAP:
                     localexp.expandTotalEndOverlap( nextIt );
                     break;
                  case MemoryChunk::SUBCHUNK_END_OVERLAP:
                     /* ERROR, it is impossible to have Subchunk End Overlap with hintNext */
                     fprintf(stderr, "error SUBCHUNK_END_OVERLAP %s\n", __FUNCTION__ );
                     break;
               }
            }
         }
      }

      void mergeNoOverlap( iterator &thisIt, const_iterator &inputIt ) {
         const MemoryChunk &thisKey = thisIt->first;
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( inputKey.getAddress() + inputKey.getLength() == thisKey.getAddress() && thisAndInputDataAreEqual ) {
            std::pair<iterator, bool> insertResult;
            inputKey.expandIncluding( thisKey );
            _thisMap.erase( thisIt );
            insertResult = _thisMap.insert( typename BaseMap::value_type( inputKey, thisData ) ); //reuse data
            thisIt = insertResult.first;
            tryToMergeWithPreviousEntry( thisIt );
            inputIt++;
         } else if ( thisKey.getAddress() > inputKey.getAddress() ) {
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *inputData ) ) );
            tryToMergeWithPreviousEntry( thisIt );
            thisIt++;
            inputIt++;
         } else {
            thisIt++;
         }
      }

      void mergeBeginOverlap( iterator &thisIt, const_iterator &inputIt ) {
         const MemoryChunk &thisKey = thisIt->first;
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( thisAndInputDataAreEqual ) {
            /* I CAN MERGE */
            std::pair<typename BaseMap::iterator, bool> insertResult;
            inputKey.expandIncluding( thisKey );
            _thisMap.erase( thisIt );
            insertResult = _thisMap.insert( typename BaseMap::value_type( inputKey, thisData ) ); //reuse data
            // insertResult.second must be true
            thisIt = insertResult.first;
         } else {
            MemoryChunk rightLeftOver;
            MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
            MemoryChunk::intersect( inputKey, thisNoConst, rightLeftOver );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *inputData ) ));
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( rightLeftOver, NEW _Type( *thisData ) ));
            thisData->merge( *inputData );
         }
         inputIt++;
      }

      void mergeEndOverlap( iterator &thisIt, const_iterator &inputIt ) {
         MemoryChunk thisKey = thisIt->first;
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         MemoryChunk rightLeftOver;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( thisAndInputDataAreEqual || thisData->contains( *inputData ) ) {
            MemoryChunk::intersect( thisKey, inputKey, rightLeftOver );
         }
         else {
            MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
            MemoryChunk::intersect( thisNoConst, inputKey, rightLeftOver );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *thisData ) ));
            thisIt->second->merge( *inputData );
            tryToMergeWithPreviousEntry( thisIt );
         }
         expand( rightLeftOver, inputData, thisIt );
         inputIt++;
      }

      void mergeTotalOverlap( iterator &thisIt, const_iterator &inputIt ) {
         MemoryChunk thisKey = thisIt->first;
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         MemoryChunk leftLeftOver;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         MemoryChunk::partition( inputKey, thisKey, leftLeftOver );
         if ( thisAndInputDataAreEqual ) {
            std::pair<typename BaseMap::iterator, bool> insertResult;
            inputKey.expandIncluding( thisKey );
            _thisMap.erase( thisIt );
            insertResult = _thisMap.insert( typename BaseMap::value_type( inputKey, thisData ) ); // do not allocate new data, reuse old
            thisIt = insertResult.first;
         } else {
            thisIt->second->merge( *inputData );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, inputData ));
            thisIt++;
         }
         expand( leftLeftOver, inputData, thisIt );
         inputIt++;
      }

      void mergeSubchunkOverlap( iterator &thisIt, const_iterator &inputIt ) {
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         MemoryChunk leftLeftOver;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( thisAndInputDataAreEqual || thisData->contains( *inputData ) ) {
            //do nothing
         } else {
            MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
            MemoryChunk::partition( thisNoConst, inputKey, leftLeftOver );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *inputData ) ) );
            thisIt->second->merge( *thisData );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( leftLeftOver, NEW _Type( *thisData ) ) );
         }
         inputIt++;
      }

      void mergeTotalBeginOverlap( iterator &thisIt, const_iterator &inputIt ) {
         MemoryChunk thisKey = thisIt->first;
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         MemoryChunk::partitionBeginAltB( thisKey, inputKey );
         if ( thisAndInputDataAreEqual ) { //do nothing
         } else {
            thisIt->second->merge( *inputData );
            tryToMergeWithPreviousEntry( thisIt );
         }
         inputKey.cutAfter( thisKey );
         expand( inputKey, inputData, thisIt );
         inputIt++;
      }

      void mergeSubchunkBeginOverlap( iterator &thisIt, const_iterator &inputIt ) {
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( thisAndInputDataAreEqual || thisData->contains( *inputData ) ) {
            //do_nothing
         } else {
            iterator thisItCopy;
            MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
            MemoryChunk::partitionBeginAgtB( thisNoConst, inputKey );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *thisData ) ) );
            thisItCopy = thisIt; thisItCopy--;
            thisData->merge( *inputData );
            tryToMergeWithPreviousEntry( thisItCopy );
         }
         inputIt++;
      }

      void mergeTotalEndOverlap( iterator &thisIt, const_iterator &inputIt ) {
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( thisAndInputDataAreEqual ) {
            std::pair<typename BaseMap::iterator, bool> insertResult;
            _thisMap.erase( thisIt );
            insertResult = _thisMap.insert( typename BaseMap::value_type( inputKey, thisData ) ); //reuse data
            thisIt = insertResult.first;
         } else {
            MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
            MemoryChunk::partitionEnd( inputKey, thisNoConst );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type ( *inputData ) ) );
            thisIt++;
            thisData->merge( *inputData );
            iterator nextIt = thisIt;
            nextIt++;
            if ( nextIt != _thisMap.end() ) {
               tryToMergeWithPreviousEntry( nextIt );
            }
         }
         thisIt++;
         inputIt++;
      }

      void mergeSubchunkEndOverlap( iterator &thisIt, const_iterator &inputIt ) {
         MemoryChunk inputKey = inputIt->first;
         _Type *inputData = inputIt->second;
         _Type *thisData = thisIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( thisAndInputDataAreEqual ) {
            //do_nothing;
         } else {
            MemoryChunk &thisNoConst = const_cast< MemoryChunk & >( thisIt->first );
            MemoryChunk::partitionEnd( thisNoConst, inputKey );
            thisIt = _thisMap.insert( thisIt, typename BaseMap::value_type( inputKey, NEW _Type( *inputData ) ) );
            thisIt->second->merge( *thisData );
         }
         thisIt++;
         inputIt++;
      }

   };
   LocalFunctions local( *this );

   const_iterator inputIt = mm.begin();
   iterator thisIt = this->begin();

   while ( inputIt != mm.end() )
   {
      if ( thisIt == this->end() ) {
         //insert the rest of the input MemoryMap, because there are no more entries in "this"
         for (; inputIt != mm.end(); inputIt++ )
         {
            thisIt = this->insert( thisIt, typename BaseMap::value_type( inputIt->first, NEW _Type ( *( inputIt->second ) ) ) );
            local.tryToMergeWithPreviousEntry( thisIt );
         }
      } else {
         MemoryChunk thisKey = thisIt->first, inputKey = inputIt->first;
         _Type *thisData = thisIt->second, *inputData = inputIt->second;
         bool thisAndInputDataAreEqual = thisData->equal( *inputData );
         if ( thisKey.equal( inputKey ) ) {
            if ( !thisAndInputDataAreEqual )
            {
               thisIt->second->merge( *inputData );
               local.tryToMergeWithPreviousEntry( thisIt );
            }
            if ( thisKey.equal( thisIt->first) ) {
               thisIt++; //otherwhise it has been merged with prev
            }
            inputIt++;
         } else {
            switch ( thisKey.checkOverlap( inputKey ) )
            {
               case MemoryChunk::NO_OVERLAP:
                  local.mergeNoOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::BEGIN_OVERLAP:
                  local.mergeBeginOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::END_OVERLAP:
                  local.mergeEndOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::TOTAL_OVERLAP:
                  local.mergeTotalOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::SUBCHUNK_OVERLAP:
                  local.mergeSubchunkOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::TOTAL_BEGIN_OVERLAP:
                  local.mergeTotalBeginOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::SUBCHUNK_BEGIN_OVERLAP:
                  local.mergeSubchunkBeginOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::TOTAL_END_OVERLAP:
                  local.mergeTotalEndOverlap( thisIt, inputIt );
                  break;
               case MemoryChunk::SUBCHUNK_END_OVERLAP:
                  local.mergeSubchunkEndOverlap( thisIt, inputIt );
                  break;
            }
         }
      }
   }
   while ( thisIt != this->end() ) {
      local.tryToMergeWithPreviousEntry( thisIt );
      thisIt++;
   }
}

} // namespace nanos

#endif /* _NANOS_MERGEABLE_MEMORYMAP_H */
