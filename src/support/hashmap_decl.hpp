/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/

#ifndef _NANOS_HASH_MAP_DECL
#define _NANOS_HASH_MAP_DECL

#define USE_NANOS_LIST 1

#include "list.hpp"
#include <list>
#include "atomic.hpp"


namespace nanos {

template<typename _KeyType>
class Hash
{
   public:
      virtual size_t operator()( _KeyType key, size_t size )
         { return ((size_t)key) % size; }
};


template <typename _KeyType, typename _T, bool _invalidate = false, size_t _tsize = 256, typename _HashFunction = Hash<_KeyType> >
class HashMap
{
   private:

      class MapEntry
      {
         private:
            _KeyType _key;
            _T _value;
            unsigned int _lru;
         public:
            MapEntry( _KeyType k ) : _key(k), _value(), _lru(0) {}
            MapEntry( _KeyType k, _T v ) : _key(k), _value(v), _lru(0) {}
            MapEntry( MapEntry const &e ) : _key(e._key), _value(e._value), _lru(0) {}
            MapEntry& operator=( MapEntry const &e )
            {
               if (this == &e) return *this;
               _key = e._key;
               _value = e._value;
               return *this;
            }
            bool operator==( MapEntry const &e ) const
            {
               return _key == e._key;
            }
            _KeyType& getKey()
               { return _key; }
            _T& getValue()
               { return _value; }
            void setLRU( unsigned int val )
               { _lru = val; }
            unsigned int getLRU()
               { return _lru; }
      };

#ifdef USE_NANOS_LIST
      typedef List<MapEntry> HashList;
#else
      typedef std::list<_T> HashList;
#endif

   public:
      class iterator {
         private:
            friend class HashMap;
            HashMap& _map;

            int _currentTable;

            typename HashList::iterator _currentItem;

            void skip()
            {
                  if ( _currentTable == -1 )
                     return;
                  while ( _currentItem == _map._table[_currentTable].end() && _currentTable + 1 < (int)_map._tableSize ) { 
                     _currentItem = _map._table[++_currentTable].begin();
                  }

                  if ( _currentItem == _map._table[_currentTable].end() ) {
                     _currentTable = -1;
                  }
            }

         public:
            iterator( HashMap &hm, int currT, typename HashList::iterator item ) : _map(hm), _currentTable( currT ), _currentItem( item )
            {
               if ( currT != -1 ) skip();
            }

            iterator( iterator const &it ) : _map( it._map ), _currentTable( it._currentTable ), _currentItem( it._currentItem ) {}

            ~iterator() {}

            iterator const& operator=( iterator const &it )
            {
               _map = it._map;
               _currentTable = it._currentTable;
               _currentItem = it._currentItem;
               return *this;
            }

            iterator operator++( int unused )
            {
               if ( _currentTable != -1 ) {
                  _currentItem++;
                  skip();
               }
               return *this;
            }

            iterator operator++()
            {
               if ( _currentTable != -1 ) {
                  _currentItem++;
                  while ( _currentTable < _map.tableSize && _currentItem == _map._table[_currentTable].end() ) {
                     _currentTable++;
                     _currentItem = _map._table[_currentTable].begin();
                  }
                  if ( _currentTable = _map._tableSize ) {
                     _currentTable = -1;
                  }
               }
               return *this;
            }

            _T const& operator*() const
               { return _currentItem->getValue() ; }

            _T& operator*()
               { return _currentItem->getValue(); }

            _T const * operator->() const
               { return &(_currentItem->getValue()); }

            _T* operator->()
               { return &(_currentItem->getValue()); }

           bool operator==( iterator const &it ) const
              { return (_currentTable == -1 && it._currentTable == -1) || _currentItem == it.currentItem; }

           bool operator!=( iterator const &it ) const
              { return !((_currentTable == -1 && it._currentTable == -1) || _currentItem == it._currentItem); }
      };

   private:
      size_t _tableSize;
      HashList _table[_tsize];
      _HashFunction _hash; 

      Atomic<unsigned int> _lruCounter;
   public:
      typedef std::list<_T> ItemList;
      typedef std::map<unsigned int, _KeyType> KeyList;

      HashMap() : _tableSize(_tsize), _table(), _hash(), _lruCounter(0) {}

      ~HashMap() {}

     /* \brief Looks up for an element in the hash, if none fouund, it creates it.
      * Thread safe if the user holds a reference on the element, the element may be deteted after the call
      */
      _T& operator[]( _KeyType key );

     /* \brief Looks up for an element in the hash
      * Thread safe if the user holds a reference on the element otherwise, the element may be deleted after the call
      */
      _T* find( _KeyType key );

     /* \brief Equivalet to operator[] but atomically increases references
      */
      _T& accessAndReference( _KeyType key );

     /* \brief Inserts atomically the given element with the given key. Sets inserted to true if so,
      *        and returns a reference to the inserted element.
      */
      _T& insert( _KeyType key, _T& elem, bool& inserted );

     /* \brief Equivalet to find but atomically increases references
      */
      _T* findAndReference( _KeyType key );

     /* \brief Decreases references
      */
      void deleteReference( _KeyType key );

     /* \brief Returns the number of references
      */
      unsigned int getReferenceCount( _KeyType key );

     /* \brief Tries to erase the element identified by 'key' and returns true if successful (it had no references)
      */
      bool erase( _KeyType key );

     /* \brief Retrns a list of keys corresponding to entries with no references
      */
      void listUnreferencedKeys( KeyList& unreferenced );

     /* \brief Removes all elements in the hash, this is not thread safe
      */
      void flush( ItemList& removedItems );

      iterator begin() {
        return iterator( *this, 0, _table[0].begin() );
      }

      iterator end() {
        return iterator( *this, -1, _table[0].begin() );
      }
};

}

#endif

