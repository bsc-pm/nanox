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

#ifndef _NANOS_COMMUTATONDEPOBJ_DECL
#define _NANOS_COMMUTATONDEPOBJ_DECL

#include "dependableobject_decl.hpp"
#include "basedependency_decl.hpp"
#include "task_reduction_decl.hpp"

namespace nanos {

  /*! \class CommutationDO
   *  \brief DependableObject used as closing dependency for commutation tasks
   */
   class CommutationDO : public DependableObject
   {
      private:
         //! Base address that determines the reduction
         BaseDependency*  _target;
         //! Flag to separate between concurrent and commutative accesses. 
         bool             _commutative; 
         TaskReduction   *_taskReduction;
      public:
        /*! \brief Default constructor
         */
         CommutationDO ( BaseDependency const& target, bool isCommutativeFlag )
            : DependableObject(), _target( target.clone() ), _commutative( isCommutativeFlag ), _taskReduction(NULL) { }

        /*! \brief Copy constructor
         *  \param cdo another CommutationDO
         */
         CommutationDO( const CommutationDO &cdo )
            : DependableObject(), _target( cdo._target->clone() ), _commutative( cdo._commutative ) { }

        /*! \brief Assignment operator
         *  \param cdo another CommutationDO
         */
         const CommutationDO & operator= ( const CommutationDO &cdo )
         {
            if ( this == &cdo ) return *this;
            DependableObject::operator= ( cdo );
            _target = cdo._target->clone();
            _commutative = cdo._commutative; 
            return *this;
         }

        /*! \brief virtual destructor
         */
         virtual ~CommutationDO()
         {
            delete _target;
         }

         /*! \brief All predecessors finished, will just execute finished and trigger its successors
          */
         virtual void dependenciesSatisfied ( );
 
         /*! \brief Returns true if this is for commutative accesses rather than concurrent ones. 
          */ 
         bool isCommutative() const; 

         void setTaskReduction( TaskReduction *tr );
   };

} // namespace nanos

#endif


