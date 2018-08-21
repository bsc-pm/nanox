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

#ifndef _NANOS_DEPENDABLE_OBJECT_WD_DECL
#define _NANOS_DEPENDABLE_OBJECT_WD_DECL

#include "synchronizedcondition_decl.hpp"
#include "dependableobject_decl.hpp"
#include "workdescriptor_fwd.hpp"

namespace nanos {

   /*! \class DOSubmit
    *  \brief DependableObject representing a WorkDescriptor as Dependable entity
    */
   class DOSubmit : public DependableObject
   {
      public:
         /*! \brief DOSubmit default constructor
          */
         DOSubmit ( ) : DependableObject() { }

         /*! \brief DOSubmit constructor
          */
         DOSubmit ( WorkDescriptor* wd) : DependableObject ( wd ) { }

         /*! \brief DOSubmit copy constructor
          *  \param dos another DOSubmit
          */
         DOSubmit ( const DOSubmit &dos ) : DependableObject( dos ) { }

         /*! \brief DOSubmit assignment operator, can be self-assigned.
          *  \param dos another DOSubmit
          */
         const DOSubmit & operator= ( const DOSubmit &dos );

         /*! \brief DOSubmit virtual destructor
          */
         virtual ~DOSubmit ( ) { }

         /*! \brief Submits WorkDescriptor when dependencies are satisfied
          */
         virtual void dependenciesSatisfied ( );
         
         /*! \brief Because of the batch-release mechanism,
          *  dependenciesSatisfied will never be called, so common code
          *  between the normal path and the batch path must go here.
          *  \see dependenciesSatisfied()
          */
         virtual void dependenciesSatisfiedNoSubmit ( );
         
         /*! \brief Checks if it has only one predecessor.
          */
         virtual bool canBeBatchReleased ( ) const;

         /*! \brief TODO 
          */
         unsigned long getDescription ( );

         /*! \brief Get the related object which actually has the dependence
          */
         virtual void * getRelatedObject ( );
         
         virtual const void * getRelatedObject ( ) const;

         /*! \brief Instrument predecessor -> successor dependency
          */
         virtual void instrument ( DependableObject& successor );
   };

  /*! \brief DependableObject representing a WorkDescriptor as a task domain to wait on some dependencies
   */
   class DOWait : public DependableObject
   {
      private:
#ifdef HAVE_NEW_GCC_ATOMIC_OPS
         bool       _depsSatisfied; /**< Condition to satisfy before execution can go forward */
#else
         volatile bool       _depsSatisfied; /**< Condition to satisfy before execution can go forward */
#endif
         SingleSyncCond<EqualConditionChecker<bool> >  _syncCond; /**< TODO */
      public:
         /*! \brief DOWait default constructor
          */
         DOWait ( ) : DependableObject(), _depsSatisfied( false ),
            _syncCond( EqualConditionChecker<bool>( &_depsSatisfied, true ) ) { }

         /*! \brief DOWait constructor
          */
         DOWait ( WorkDescriptor *wd ) : DependableObject( wd ), _depsSatisfied( false ),
           _syncCond( EqualConditionChecker<bool>( &_depsSatisfied, true ) ) { }

         /*! \brief DOWait copy constructor
          *  \param dos another DOWait
          */
         DOWait ( const DOWait &dow ) : DependableObject(dow), _depsSatisfied( false ),
           _syncCond( EqualConditionChecker<bool>( &_depsSatisfied, true ) ) { }

         /*! \brief DOWait assignment operator, can be self-assigned.
          *  param dos another DOWait
          */
         const DOWait & operator= ( const DOWait &dow );

         /*! \brief Virtual destructor
          */
         virtual ~DOWait ( ) { }

         /*! \brief Initialise wait condition
          */
         virtual void init ( );

         /*! \brief whether the DO gets blocked and no more dependencies can
          *  be submitted until it is satisfied.
          */
         virtual bool waits ( );

         /*! \brief Unblock method when dependencies are satisfied
          */
         virtual void dependenciesSatisfied ( );

         /*! \brief
          */
         int decreasePredecessors ( std::list<uint64_t>const * flushDeps,  DependableObject * finishedPred,
               bool batchRelease, bool blocking = false );

         /*! \brief TODO
          */
         //void setWD( WorkDescriptor *wd );

         /*! \brief Get the related object which actually has the dependence
          */
         virtual void * getRelatedObject ( );
         
         virtual const void * getRelatedObject ( ) const;

         /*! \brief Instrument predecessor -> successor dependency
          */
         virtual void instrument ( DependableObject& successor );
   };

} // namespace nanos

#endif

