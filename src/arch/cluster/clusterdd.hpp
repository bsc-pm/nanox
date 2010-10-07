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

#ifndef _NANOS_CLUSTER_WD
#define _NANOS_CLUSTER_WD

#include "config.hpp"
#include "smpdd.hpp"
#include "clusterdevice.hpp"
#include "workdescriptor.hpp"

namespace nanos {
namespace ext
{

   extern ClusterDevice Cluster;

   class ClusterPlugin;
   class ClusterDD : public SMPDD
   {
      friend class ClusterPlugin;

      public:
         // constructors
         ClusterDD( work_fct w ) : SMPDD( w, &SMP ) {}

         ClusterDD() : SMPDD( &SMP ) {}

         // copy constructors
         ClusterDD( const ClusterDD &dd ) : SMPDD( dd ) {}

         // assignment operator
         const ClusterDD & operator= ( const ClusterDD &wd );

         // destructor
         virtual ~ClusterDD() { }
   };

   inline const ClusterDD & ClusterDD::operator= ( const ClusterDD &dd )
   {
      // self-assignment: ok
      if ( &dd == this ) return *this;

      DD::operator= ( dd );

      return *this;
   }
}
}

#endif
