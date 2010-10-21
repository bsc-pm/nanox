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

#ifndef _NANOS_LIST
#define _NANOS_LIST

#include "atomic.hpp"
#include <list>
#include <limits.h>
#include <iterator>
#include <stddef.h>

namespace nanos {

template<class _T>
class List {
   public:

      class ListNode {
         private:
            _T             _object;
            ListNode*     _next;
            Atomic<int>   _refs;
            volatile bool _valid;
         public:
            ListNode() : _object(), _next(NULL), _refs(0), _valid(false) {}

            ListNode( _T &object, ListNode* next = NULL ) : _object(object), _next(next), _refs(0), _valid(false) {}

            ListNode( const _T &object, ListNode* next = NULL ) : _object(object), _next(next), _refs(0), _valid(false) {}

            ListNode( ListNode const &node ) : _object(node._object), _next(node._next), _refs(node._refs), _valid(node._valid) {}

            ListNode& operator=( ListNode const &node)
            {
               if (this == &node) return *this;
               this._object = node._object;
               this._next = node._next;
               return *this;
            }

            const _T& getItem() const
               { return _object; }

            _T& getItem()
               { return _object; }

            void setItem( _T& item )
               { _object = item; }

            void setItem( const _T& item )
               { _object = item; }

            ListNode* next() const
               { return _next; }

            void setNext( ListNode* next )
               { _next = next; }

            bool hasRefs() const
               { return _refs > 0; }

            int getRefs() const
               { return _refs.value(); }

            void setRefs( int refs )
               { _refs = refs; }

            int increaseRefs()
               { return _refs++; }

            int decreaseRefs()
               { return ( _refs.value() <= 0 ? _refs.value() : _refs--); }

            bool setRefsCswap( int val )
            {
               Atomic<int> putVal = val;
//               Atomic<int> expected = 0;
               Atomic<int> expected = 1; // the last reference will always be the iterator itself
               return _refs.cswap( expected, putVal );
            }

            bool isValid() const
               { return _valid; }

            void setValid( bool valid )
               { _valid = valid; }
      };

      class const_iterator;

      class iterator {
         private:
            ListNode*   _node;
            List*       _list;

            friend class List;
            friend class const_iterator;
         public:
            typedef std::forward_iterator_tag iterator_category;
            typedef _T                   value_type;
            typedef ptrdiff_t            difference_type;
            typedef _T*                  pointer;
            typedef _T&                  reference;

            iterator( ListNode* node, List* list ) : _node(node), _list(list) { }

            iterator( iterator const &it ) : _node(it._node), _list(it._list)
            {
               if ( _node != _list->_end ) {
                  _node->increaseRefs();
               }
            }

            ~iterator()
            {
               if ( _node != _list->_end ) {
                  _node->decreaseRefs();
               }
            }

            operator const_iterator () const
               { return const_iterator( _node, _list ); }

            iterator const& operator=( iterator const &it )
            {
               if ( _node != _list->_end )
                  _node->decreaseRefs();
               _node = it._node;
               _list = it._list;
               if ( _node != _list->_end )
                  _node->increaseRefs();
               return *this;
            }

            void skip()
            {
               while ( _node != _list->_end ) {
                  if ( _node->isValid() ) {
                     if ( _node->increaseRefs() >= 0 ) {
                        if ( _node->isValid() ) {
                           return;
                        }
                     }
                     _node->decreaseRefs();
                  }
                  _node = _node->next();
               }
            }

            iterator operator++( int unused )
            {
               iterator result = iterator(*this);
               if ( _node != _list->_end ) {
                  _node->decreaseRefs();
                  _node = _node->next();
                  if ( _node != _list->_end ) {
                     skip();
                  }
               }
               return result;
            }

            iterator operator++()
            {
               if ( _node != _list->_end ) {
                  _node->decreaseRefs();
                  _node = _node->next();
                  if ( _node != _list->_end ) {
                     skip();
                  }
               }
               return *this;
            }

           _T const& operator*() const
              { return _node->getItem(); } 

           _T& operator*()
              { return _node->getItem(); }

           _T const * operator->() const
              { return &(_node->getItem()); }

           _T* operator->()
              { return &(_node->getItem()); }

           bool operator==( iterator const &it ) const
              { return _node == it._node; }

           bool operator!=( iterator const &it ) const
              { return _node != it._node; }

           bool operator==( const_iterator const &it ) const
              { return _node == it._node; }

           bool operator!=( const_iterator const &it ) const
              { return _node != it._node; }

           int getReferences() const
           {
              if (_node != _list->_end ) {
                 return _node->getRefs();
              }
              return 0;
           }

           void addReference()
           {
              if ( _node != _list->_end ) {
                 _node->increaseRefs();
              }
           }

           void deleteReference()
           {
              if ( _node != _list->_end ) {
                 _node->decreaseRefs();
              }
           }
      };

      class const_iterator {
         private:
            ListNode*   _node;
            List*       _list;

            friend class List;
            friend class iterator;

         public:
            typedef std::forward_iterator_tag iterator_category;
            typedef _T                   value_type;
            typedef ptrdiff_t            difference_type;
            typedef _T*                  pointer;

            const_iterator( ListNode* node, List* list ) : _node(node), _list(list) {}

            const_iterator( const_iterator const &it ) : _node(it._node), _list(it._list) {}

            ~const_iterator()
            {
               if ( _node != _list->_end )
                  _node->decreaseRefs();
            }

            operator const_iterator () const
               { return const_iterator( _node, _list ); }

            void skip()
            {
               while ( _node != _list->_end ) {
                  if ( _node->isValid() ) {
                     if ( _node->increaseRefs() >= 0 ) {
                        if ( _node->isValid() ) {
                           return;
                        }
                     }
                     _node->decreaseRefs();
                  }
                  _node = _node->next();
               }
            }

            const_iterator operator++( int unused )
            {
               const_iterator result = const_iterator(*this);
               if ( _node != _list->_end ) {
                  _node->decreaseRefs();
                  _node = _node->next();
                  if ( _node != _list->_end ) {
                     skip();
                  }
               }
               return result;
            }

            const_iterator operator++()
            {
               if ( _node != _list->_end ) {
                  _node->decreaseRefs();
                  _node = _node->next();
                  if ( _node != _list->_end ) {
                     skip();
                  }
               }
               return *this;
            }

           _T const& operator*() const
              { return _node->getItem(); } 

           _T& operator*()
              { return _node->getItem(); }

           _T const * operator->() const
              { return &(_node->getItem()); }

           _T* operator->()
              { return &(_node->getItem()); }

           bool operator==( const_iterator const &it ) const
              { return _node == it._node; }

           bool operator!=( const_iterator const &it ) const
              { return _node != it._node; }

           bool operator==( iterator const &it ) const
              { return _node == it._node; }

           bool operator!=( iterator const &it ) const
              { return _node != it._node; }

           int getReferences() const
           {
              if (_node != _list->_end ) {
                 return _node->getRefs();
              }
              return 0;
           }

           void addReference()
           {
              if ( _node != _list->_end ) {
                 _node->increaseRefs();
              }
           }

           void deleteReference()
           {
              if ( _node != _list._end ) {
                 _node->decreaseRefs();
              }
           }
      };

   public:
      typedef std::list<ListNode*> NodeList;

   private:
      NodeList _freeList;
      // FIXME: eventually this will be static
      int _N;

      Lock _lock;

   protected:
      ListNode* _begin;
      ListNode* _end;
      size_t _size;

   public:
      List() : _freeList(), _N(INT_MIN), _lock(), _begin(NULL), _end(NULL), _size(0) {}

      ~List()
      {
         ListNode *it = _begin;
         while ( it != _end ) {
            ListNode* tmp = it;
            it = it->next();
            delete tmp;
         }
      }

      _T& front()
         { return *_begin->getItem(); }

      _T& push_front( const _T& elem )
      {
         ListNode *node;
         if (_freeList.empty() ) {
            node = new ListNode( elem );
            node->setNext( _begin );
            _begin = node;
            node->setValid( true );
         } else {
            node = _freeList.front();
            _freeList.pop_front();
            node->setItem( elem );
            node->setRefs(0);
            node->setValid( true );
         }
         _size++;
         return node->getItem();
      }

      void pop_front()
      {
         // Get an iterator to the first valid element
         iterator it = begin();

         while ( it != end() ) {
            // Invalidate the element
            ListNode *elem = it._node;
            if ( elem->setRefsCswap(_N) ) {
               elem->setValid(false);
               _freeList.push_back(elem);
               _size--;
               return;
            }
            it++;
         }
      }

      bool erase( const _T& t )
      {
         iterator it = begin();
         while ( it!= end()) {
            if ( t == *it ) {
               ListNode *elem = it._node;
               if ( elem->setRefsCswap(_N) ) {
                  elem->setValid(false);
                  _freeList.push_back(elem);
                  _size--;
                  return true;
               }
            return false;
            }
            it++;
         }
         return false;
      }

      iterator begin()
      {
         iterator it(_begin, this);
         it.skip();
         return it;
      }

      iterator end()
      {
         return iterator(_end, this);
      }

      size_t size() const
      {
         return _size;
      }

     /*! \brief Get exclusive access to the list
      */
      void lock ( )
      {
         _lock.acquire();
         memoryFence();
      }

     /*! \brief Release list's lock
      */
      void unlock ( )
      {
         memoryFence();
         _lock.release();
      }
};

}
#endif

