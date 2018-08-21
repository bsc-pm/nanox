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

#ifndef _CLUSTERNODE_DECL
#define _CLUSTERNODE_DECL

#include "config.hpp"
#include "simpleallocator_decl.hpp"
#include "clusterdevice_decl.hpp"
#include "smpdd.hpp"
#ifdef GPU_DEV
//FIXME: GPU support
#include "gpudd.hpp"
#endif
#include "cachedaccelerator_decl.hpp"

namespace nanos {
namespace ext {

      class ClusterNode : public ProcessingElement
      {
         public:
            typedef std::map<unsigned int, const Device *> ClusterSupportedArchMap;

         private:
            // config variables
            static Atomic<int>      _deviceSeed; // Number of cluster devices assigned to threads
            unsigned int            _clusterNode; // Assigned cluster device Id
            unsigned int _executedWorkDesciptors;
            ClusterSupportedArchMap _supportedArchsById;

            // disable copy constructor and assignment operator
            ClusterNode( const ClusterNode &pe );
            const ClusterNode & operator= ( const ClusterNode &pe );

         public:
            // constructors
            ClusterNode( int nodeId, memory_space_id_t memId,
               ClusterSupportedArchMap const &archs, const Device **archsArray );
            virtual ~ClusterNode();

            virtual WD & getWorkerWD () const;
            virtual WD & getMasterWD () const;
            virtual WD & getMultiWorkerWD () const;
            virtual BaseThread & createThread ( WorkDescriptor &wd, SMPMultiThread *parent );
            virtual BaseThread & createMultiThread ( WorkDescriptor &wd, unsigned int numPEs, PE **repPEs );

            // capability query functions
            virtual bool supportsUserLevelThreads () const;
            virtual unsigned int getMyNodeNumber() const;

            unsigned int getClusterNodeNum() const;
            ClusterSupportedArchMap const &getSupportedArchs() const;
            SimpleAllocator & getAllocator( void );

            void incExecutedWDs();
            unsigned int getExecutedWDs() const;
            unsigned int getNodeNum() const;

            static void clusterWorker();
      };
} // namespace ext
} // namespace nanos


#endif /* _CLUSTERNODE_DECL */
