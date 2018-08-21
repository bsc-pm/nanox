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

#ifndef _NANOS_TASK_REDUCTION_DECL_H
#define _NANOS_TASK_REDUCTION_DECL_H

#include "nanos-int.h"

//! \brief This class represent a Task Reduction.
//!
//! It contains all the information needed to handle a task reduction. It storages the
//! thread private copies and also keep all the information in order to compute final
//! reduction: reducers.
//

namespace nanos {

class TaskReduction {

   public:

      typedef void ( *initializer_t ) ( void *omp_priv,  void* omp_orig );
      typedef void ( *reducer_t ) ( void *obj1, void *obj2 );
      typedef struct {void * data; bool isInitialized;} field_t;
      typedef std::vector<field_t> storage_t;




   private:

      // These two variables have the same value in almost all the cases. They
      // are only different when we are doing a Fortran Array Reduction
      void           *_original;         //!< Original variable address
      void           *_dependence;       //!< Related dependence

      unsigned        _depth;            //!< Reduction depth
      initializer_t   _initializer;      //!< Initialization function

      // These two variables have the same value in almost all the cases. They
      // are only different when we are doing a Fortran Array Reduction
      reducer_t       _reducer;          //!< Reducer operator
      reducer_t       _reducer_orig_var; //!< Reducer on orignal variable

      storage_t       _storage;          //!< Private copy vector
      size_t          _size;             //!< Size of array (size of element is scalar)
      size_t          _size_element;     //!< Size of element
      size_t          _num_elements;     //!< Number of elements (for a scalar reduction, this is 1)
      size_t          _num_threads;      //!< Number of threads (private copies)
      void           *_min;              //!< Pointer to first private copy
      void           *_max;              //!< Pointer to last private copy
      bool            _isLazyPriv;       //!< Is lazy privatization enabled
      bool            _isFortranArrayReduction;//!< whether this is a Fortran array reudction

      //! \brief TaskReduction copy constructor (disabled)
      TaskReduction( const TaskReduction &tr ) {}

   public:

      //! \brief TaskReduction constructor only used when we are performing a Reduction
      TaskReduction( void *orig, initializer_t f_init, reducer_t f_red,
    		  	  size_t size, size_t size_elem, size_t
				  threads, unsigned depth, bool lazy )
               	   : _original(orig), _dependence(orig), _depth(depth), _initializer(f_init),
					 _reducer(f_red), _reducer_orig_var(f_red), _storage(threads),
					 _size(size), _size_element(size_elem),_num_elements(size/size_elem),
					 _num_threads(threads), _min(NULL), _max(NULL), _isLazyPriv (lazy), _isFortranArrayReduction(false)
   {
      if(_isLazyPriv) {
         //Note that renaming tracking for nested reductions is not supported
         //for lazy privatization (_min = _max = NULL)

         for ( size_t i=0; i<_num_threads; i++) {
            _storage[i].data = NULL;
            _storage[i].isInitialized = false;
         }
      }
      else {
         NANOS_ARCHITECTURE_PADDING_SIZE(_size);

         char * storage = (char*) malloc (_size*threads);
         _min = & storage[0];
         _max = & storage[_size * threads];
         for ( size_t i=0; i<_num_threads; i++) {
            _storage[i].data = (void *) &storage[i * _size];
            _storage[i].isInitialized = false;
         }
      }
   }

      //!brief TaskReduction constructor only used when we are performing a Fortran Array Reduction
   TaskReduction( void *orig, void *dep, initializer_t f_init, reducer_t f_red,
            reducer_t f_red_orig_var, size_t array_descriptor_size, size_t
            threads, unsigned depth, bool lazy )
         : _original(orig), _dependence(dep), _depth(depth),
         _initializer(f_init), _reducer(f_red), _reducer_orig_var(f_red_orig_var), _storage(threads),
         _size(array_descriptor_size), _size_element(0),_num_elements(0),
         _num_threads(threads), _min(NULL), _max(NULL), _isLazyPriv(lazy), _isFortranArrayReduction(true)
   {

      if(_isLazyPriv) {
         //Note that renaming tracking for nested reductions is not supported
         //for lazy privatization (_min = _max = NULL)

         for ( size_t i=0; i<_num_threads; i++) {
            _storage[i].data = NULL;
            _storage[i].isInitialized = false;
         }
      }
      else {
         NANOS_ARCHITECTURE_PADDING_SIZE(_size);
         char * storage = (char*) malloc (_size * threads);

         _min = & storage[0];
         _max = & storage[_size * threads];
         for ( size_t i=0; i<_num_threads; i++) {
            _storage[i].data = (void *) &storage[i * _size];
            _storage[i].isInitialized = false;
         }
      }
   }

      //! \brief Taskreduction destructor
      ~TaskReduction() {
         if(_isLazyPriv) {
            for ( size_t i = 0; i < _num_threads; i++) {
               free(_storage[i].data);
            }
         }
         else {
            free(_storage[0].data);
         }
      }

      //! \brief
      //! \smart text here
      bool has ( const void *ptr );

      //! \brief
      //! \smart text here
      void * get ( size_t id );

      //! \brief This function reduces the content of the private copies to the
      //original one. Currently, it also re-initializes to the neutral element
      //these private copies because we cannot guarantee that the reduction has
      //been finalized
      void reduce();

      //! \brief It allocates the private copy associated with the 'id' thread
      void * allocate( size_t id );

      //! \brief It initializes the private copy associated with the 'id' thread
      void initialize( size_t id );

      //! \brief Get depth where task reduction were registered
      unsigned getDepth( void ) const;

      bool isInitialized( size_t id );
};

} // namespace nanos

#endif
