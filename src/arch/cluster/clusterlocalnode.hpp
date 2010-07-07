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

#ifndef _NANOS_CLUSTER_LOCAL_NODE
#define _NANOS_CLUSTER_LOCAL_NODE

#include "clusternode.hpp"
#include "config.hpp"
#include "clusterdevice.hpp"
#include "clusterthread.hpp"

namespace nanos {
namespace ext
{

   class ClusterLocalNode : public ClusterNode
   {


      private:
         // disable copy constructor and assignment operator
         ClusterLocalNode( const ClusterLocalNode &pe );
         const ClusterLocalNode & operator= ( const ClusterLocalNode &pe );

         //Cache<ClusterDevice> _cache;

      public:
         // constructors
         ClusterLocalNode( int id ) : ClusterNode( id ) { }

         virtual ~ClusterLocalNode() {}

         void stopAll();
         //{
         //    fprintf(stderr, "stoping local threads.\n");
         //}

         //virtual WD & getWorkerWD () const;
         //virtual WD & getMasterWD () const;
         //virtual BaseThread & createThread ( WorkDescriptor &wd );

         //// capability query functions
         //virtual bool supportsUserLevelThreads () const { return false; }

         ///* Memory space support */
         //virtual void registerDataAccessDependent( uint64_t tag, size_t size );
         //virtual void copyDataDependent( uint64_t tag, size_t size );
         //virtual void unregisterDataAccessDependent( uint64_t tag );
         //virtual void copyBackDependent( uint64_t tag, size_t size );
         //virtual void* getAddressDependent( uint64_t tag );
         //virtual void copyToDependent( void *dst, uint64_t tag, size_t size );

         //void registerCacheAccessDependent(uint64_t a, size_t aa, bool aaa, bool aaaa);
         //void unregisterCacheAccessDependent(uint64_t a, size_t aa);
         //void registerPrivateAccessDependent(uint64_t a, size_t aa, bool aaa, bool aaaa);
         //void unregisterPrivateAccessDependent(uint64_t a, size_t aa);

   };

}
}

#endif
