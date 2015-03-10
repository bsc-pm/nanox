/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

#ifndef _NANOS_TASK_REDUCTION_DECL_H
#define _NANOS_TASK_REDUCTION_DECL_H

//! \brief This class represent a Task Reduction.
//!
//! It contains all the information needed to handle a task reduction. It storages the
//! thread private copies and also keep all the information in order to compute final
//! reduction: reducers.
class TaskReduction {
   public:
      typedef void ( *initializer_t ) ( void *omp_priv,  void* omp_orig );
      typedef void ( *reducer_t ) ( void *obj1, void *obj2 );
      typedef std::vector<char> storage_t;
   private:
      void                   *_original;         //!< Original variable
      void                   *_dependence;       //!< Related dependence
      unsigned                _depth;            //!< Reduction depth
      initializer_t           _initializer;      //!< Initialization function
      reducer_t               _reducer;          //!< Reducer opeartor
      reducer_t               _reducer_orig_var; //!< Reducer on orignal variable
      storage_t               _storage;          //!< Private copy vector
      size_t                  _size;             //!< Size of element
      size_t                  _threads;          //!< Number of threads (private copies)
      void                   *_min;              //!< Pointer to first private copy
      void                   *_max;              //!< Pointer to last private copy
   private:
      //! \brief TaskReduction copy constructor (disabled)
      TaskReduction ( const TaskReduction &tr ) {}
   public:
      //! \brief TaskReduction constructor
      TaskReduction ( void *orig, void *dep, initializer_t init, reducer_t red,
                      reducer_t red_orig_var, size_t size, size_t threads, unsigned depth )
                    : _original(orig), _dependence(dep), _depth(depth), _initializer(init),
                      _reducer(red), _reducer_orig_var(red_orig_var), _storage(size*threads),
                      _size(size), _threads(threads), _min(NULL), _max(NULL)
      {
         _min = & _storage[0];
         _max = & _storage[_size*threads];

         for ( size_t i=0; i<threads; i++) {
             _initializer( &_storage[i*_size], _original );
         }
      }
      //! \brief Taskreduction destructor
     ~TaskReduction ( )
      {
      }
      //! \brief Is the provided address the original symbol or one of the private copies
      //! \return NULL if not matches, id's corresponding private copy if matches
      void * have ( const void *ptr, size_t id ) ;
      //! \brief Is the provided address the dependence or one of the private copies
      //! \return NULL if not matches, id's corresponding private copy if matches
      void * have_dependence ( const void *ptr, size_t id ) ;
      //! \brief Finalizes reduction
      void * finalize ( void );
      //! \brief Get depth where task reduction were registered 
      unsigned getDepth(void) const ;
};

#endif
