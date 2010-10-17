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

#ifndef _NANOS_WORK_GROUP
#define _NANOS_WORK_GROUP

#include <vector>
#include "atomic.hpp"
#include "dependenciesdomain.hpp"
#include "synchronizedcondition_decl.hpp"

namespace nanos
{

   class WorkGroup
   {

      private:
         static Atomic<int> _atomicSeed;

         // FIX-ME: vector is not a safe-class here
         typedef std::vector<WorkGroup *> WGList;

         WGList         _partOf;
         int            _id;
         Atomic<int>    _components;
         Atomic<int>    _phaseCounter;

         SingleSyncCond<EqualConditionChecker<int> > _syncCond;

         void addToGroup ( WorkGroup &parent );
         void exitWork ( WorkGroup &work );

         const WorkGroup & operator= ( const WorkGroup &wg );

      public:
         // constructors
         WorkGroup() : _id( _atomicSeed++ ),_components( 0 ), _phaseCounter( 0 ),
            _syncCond( EqualConditionChecker<int>( &_components.override(), 0 ) ) {  }
         WorkGroup( const WorkGroup &wg ) : _id( _atomicSeed++ ),_components( 0 ), _phaseCounter( 0 ),
            _syncCond( EqualConditionChecker<int>(&_components.override(), 0 ) ) 
         {
            for ( WGList::const_iterator it = wg._partOf.begin(); it < wg._partOf.end(); it++ ) {
               if (*it) (*it)->addWork( *this );
            }
         }

         // destructor
         virtual ~WorkGroup();

         void addWork( WorkGroup &wg );
         void sync();
         void waitCompletion();
         virtual void init();
         virtual void done();
         int getId() const { return _id; }

   };

   typedef WorkGroup WG;

};

#endif

