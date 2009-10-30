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

namespace nanos
{

   class WorkGroup
   {

      private:
         static Atomic<int> atomicSeed;

         typedef std::vector<WorkGroup *> ListOfWGs;
         ListOfWGs partOf;
         int id;

         Atomic<int>  components;
         Atomic<int>  phase_counter;

         void addToGroup ( WorkGroup &parent );
         void exitWork ( WorkGroup &work );

      public:
         // constructors
         WorkGroup() : id( atomicSeed++ ),components( 0 ), phase_counter( 0 ) {  }

         // to do these two properly we would need to keep also the information of the components
         // TODO:copy constructor
         WorkGroup( const WorkGroup &wg );
         // TODO:assignment operator
         const WorkGroup & operator= ( const WorkGroup &wg );

         // destructor
         virtual ~WorkGroup();

         void addWork( WorkGroup &wg );
         void sync();
         void waitCompletation();
         void done();
         int getId() const { return id; }
   };

   typedef WorkGroup WG;

};

#endif

