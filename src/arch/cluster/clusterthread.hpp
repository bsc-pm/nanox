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

#ifndef _NANOS_GPU_THREAD
#define _NANOS_GPU_THREAD

#include "clusterdd.hpp"
#include "smpthread.hpp"
#include "wddeque.hpp"


namespace nanos {
namespace ext
{

   class ClusterThread : public SMPThread
   {

      friend class ClusterProcessor;

      private:
         unsigned int                     _clusterNode; // Assigned Cluster device Id
         WDDeque                          _myWDs;

         // disable copy constructor and assignment operator
         ClusterThread( const ClusterThread &th );
         const ClusterThread & operator= ( const ClusterThread &th );

      public:
         // constructor
         ClusterThread( WD &w, PE *pe, int device ) : SMPThread( w, pe ), _clusterNode( device ) {}

         // destructor
         virtual ~ClusterThread() {}

         virtual void runDependent ( void );
         virtual void inlineWorkDependent ( WD &wd );
         void addWD( WorkDescriptor *wd );
         WorkDescriptor *getWD();

   };


}
}

#endif
