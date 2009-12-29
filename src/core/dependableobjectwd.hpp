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

#ifndef _NANOS_DEPENDABLE_OBJECT_WD
#define _NANOS_DEPENDABLE_OBJECT_WD

#include "synchronizedcondition.hpp"
#include "dependableobject.hpp"

namespace nanos
{

   class WorkDescriptor;

  /*! \brief DependableObject representing a WorkDescriptor as Dependable entity
   */
   class DOSubmit : public DependableObject
   {
      private:
         /**< Pointer to the work descriptor represented by this DependableObject */
         WorkDescriptor *_submittedWD;

        /*! Disable default constructor
         */
         DOSubmit ( );
   
      public:
    
        /*! \brief Constructor
         */
         DOSubmit ( WorkDescriptor* wd) : DependableObject ( ), _submittedWD( wd ) { }
   
        /*! \brief Copy constructor
         *  \param dos another DOSubmit
         */
         DOSubmit ( const DOSubmit &dos ) : DependableObject(dos), _submittedWD( dos._submittedWD ) { } 
   
        /*! \brief Assign operator, can be self-assigned.
         *  \param dos another DOSubmit
         */
         const DOSubmit & operator= ( const DOSubmit &dos )
         {
            if ( this == &dos ) return *this; 
            DependableObject::operator= (dos);
            _submittedWD = dos._submittedWD;
            return *this;
         }
   
        /*! \brief Virtual destructor
         */
         virtual ~DOSubmit ( ) { }

         virtual void dependenciesSatisfied ( );
         
   };

  /*! \brief DependableObject representing a WorkDescriptor as a task domain to wait on some dependencies
   */
   class DOWait : public DependableObject
   {
      private:
        /**< Pointer to the WorkDescriptor that waits on data */
         WorkDescriptor *_waitDomainWD;

        /**< Condition to satisfy before execution can go forward */
         volatile bool _depsSatisfied;
         
         SingleSyncCond _syncCond;

        /*! Disable default constructor
         */
         DOWait ( );
   
      public:
        /*! \brief Constructor
         */
         DOWait ( WorkDescriptor *wd ) : DependableObject(), _waitDomainWD( wd ), _depsSatisfied( false ),
           _syncCond( new EqualConditionChecker<bool>( &_depsSatisfied, true ) ) { }
    
        /*! \brief Copy constructor
         *  \param dos another DOWait
         */
         DOWait ( const DOWait &dow ) : DependableObject(dow), _waitDomainWD( dow._waitDomainWD ), _depsSatisfied( false ),
           _syncCond( new EqualConditionChecker<bool>( &_depsSatisfied, true ) ) { }
   
        /*! \brief Assign operator, can be self-assigned.
         *  param dos another DOWait
         */
         const DOWait & operator= ( const DOWait &dow )
         {
            if ( this == &dow ) return *this; 
            DependableObject::operator= ( dow );
            _depsSatisfied = dow._depsSatisfied;
            return *this;
         }
   
        /*! \brief Virtual destructor
         */
         virtual ~DOWait ( ) { }

         virtual void init ( );

         virtual void wait ( );

         virtual bool waits ( );

         virtual void dependenciesSatisfied ( );
        
   };
};

#endif

