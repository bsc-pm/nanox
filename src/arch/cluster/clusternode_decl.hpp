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

#ifndef _CLUSTERNODE_DECL
#define _CLUSTERNODE_DECL

#include "config.hpp"
#include "simpleallocator_decl.hpp"
#include "clusterinfo_decl.hpp"
#include "clusterdevice_decl.hpp"
#include "smpdd.hpp"
#ifdef GPU_DEV
//FIXME: GPU support
#include "gpudd.hpp"
#endif
#include "cachedaccelerator_decl.hpp"

namespace nanos {
   namespace ext {

      class ClusterNode : public CachedAccelerator
      {

         private:
            // config variables
            static Atomic<int>      _deviceSeed; // Number of cluster devices assigned to threads
            unsigned int            _clusterNode; // Assigned cluster device Id

            // disable copy constructor and assignment operator
            ClusterNode( const ClusterNode &pe );
            const ClusterNode & operator= ( const ClusterNode &pe );

            SimpleAllocator _memSegment;
            unsigned int _executedWorkDesciptors;

         public:
            // constructors
            ClusterNode( int id, memory_space_id_t memId );

            virtual ~ClusterNode();

            virtual WD & getWorkerWD () const;
            virtual WD & getMasterWD () const;
            virtual WD & getMultiWorkerWD () const;
            virtual BaseThread & createThread ( WorkDescriptor &wd, SMPMultiThread *parent );
            virtual BaseThread & createMultiThread ( WorkDescriptor &wd, unsigned int numPEs, PE **repPEs );

            // capability query functions
            virtual bool supportsUserLevelThreads () const;
            virtual bool isGPU () const;
            //virtual bool supportsDirectTransfersWith( ProcessingElement const &pe ) const;
            virtual unsigned int getMyNodeNumber() const;

            unsigned int getClusterNodeNum() const;
            SimpleAllocator & getAllocator( void );

            void incExecutedWDs();
            unsigned int getExecutedWDs() const;
            unsigned int getNodeNum() const;
      };
   }
}

#endif /* _CLUSTERNODE_DECL */
