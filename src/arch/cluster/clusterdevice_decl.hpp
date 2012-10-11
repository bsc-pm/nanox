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

#ifndef _CLUSTER_DEVICE
#define _CLUSTER_DEVICE

#include "workdescriptor_decl.hpp"
#include "copydescriptor_decl.hpp"
#include "packer_decl.hpp"

namespace nanos
{
   namespace ext
   {
/* \brief Device specialization for cluster architecture
 * provides functions to allocate and copy data in the device
 */
   class ClusterDevice : public Device
   {
         Packer _packer;
      public:

         struct GetRequest {
            volatile int _complete;
            char* _hostAddr;
            std::size_t _size;
            char* _recvAddr;
            DeviceOps *_ops;
            Packer *_packer;

            GetRequest( char* hostAddr, std::size_t size, char *recvAddr, DeviceOps *ops ) :
               _complete(0), _hostAddr( hostAddr ), _size( size ), _recvAddr( recvAddr ), _ops( ops ) { }
            virtual ~GetRequest() {}

            void complete();
            bool isCompleted() const;
            virtual void clear();
         };

         struct GetRequestStrided : public GetRequest {
            std::size_t _count;
            std::size_t _ld;
            Packer *_packer;

            GetRequestStrided( char* hostAddr, std::size_t size, std::size_t count, std::size_t ld, char *recvAddr, DeviceOps *ops, Packer *packer ) :
               GetRequest( hostAddr, size, recvAddr, ops ), _count( count ), _ld( ld ), _packer( packer ) { }
            virtual ~GetRequestStrided() {}

            virtual void clear();
         };


         /*! \brief ClusterDevice constructor
          */
         ClusterDevice ( const char *n ) : Device ( n ) {}

         /*! \brief ClusterDevice copy constructor
          */
         ClusterDevice ( const ClusterDevice &arch ) : Device ( arch ) {}

         /*! \brief ClusterDevice destructor
          */
         ~ClusterDevice() { }

         virtual void *memAllocate( std::size_t size, ProcessingElement &pe) const;
         virtual void _copyIn( uint64_t devAddr, uint64_t hostAddr, std::size_t len, ProcessingElement const &pe, DeviceOps *ops, WD const &wd ) const;
         virtual void _copyOut( uint64_t hostAddr, uint64_t devAddr, std::size_t len, ProcessingElement const &pe, DeviceOps *ops, WD const &wd ) const;
         virtual void _copyDevToDev( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, ProcessingElement const &peDest, ProcessingElement const &peOrig, DeviceOps *ops, WD const &wd ) const;
         virtual void _copyInStrided1D( uint64_t devAddr, uint64_t hostAddr, std::size_t len, std::size_t count, std::size_t ld, ProcessingElement const &pe, DeviceOps *ops, WD const &wd ) ;
         virtual void _copyOutStrided1D( uint64_t hostAddr, uint64_t devAddr, std::size_t len, std::size_t count, std::size_t ld, ProcessingElement const &pe, DeviceOps *ops, WD const &wd ) ;
         virtual void _copyDevToDevStrided1D( uint64_t devDestAddr, uint64_t devOrigAddr, std::size_t len, std::size_t count, std::size_t ld, ProcessingElement const &peDest, ProcessingElement const &peOri, DeviceOps *ops, WD const &wd ) const;
   };

   extern ClusterDevice Cluster;

}
}


#endif
