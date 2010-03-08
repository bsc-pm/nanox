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

#ifndef _NANOS_PROCESSING_ELEMENT
#define _NANOS_PROCESSING_ELEMENT

#include "workdescriptor.hpp"
#include <algorithm>
#include "functors.hpp"

namespace nanos
{

// forward definitions

   class SchedulingGroup;
   class BaseThread;

   class ProcessingElement
   {

      private:
         typedef std::vector<BaseThread *>    ThreadList;
         int                                  _id;
         const Device *                       _device;
         ThreadList                           _threads;

         ProcessingElement ( const ProcessingElement &pe );
         const ProcessingElement & operator= ( const ProcessingElement &pe );
         
      protected:
         virtual WorkDescriptor & getMasterWD () const = 0;
         virtual WorkDescriptor & getWorkerWD () const = 0;

      public:
         // constructors
         ProcessingElement ( int newId, const Device *arch ) : _id ( newId ), _device ( arch ) {}

         // destructor
         virtual ~ProcessingElement()
         {
            std::for_each(_threads.begin(),_threads.end(),deleter<BaseThread>);
         }

         /* get/put methods */
         int getId() const {
            return _id;
         }

         const Device & getDeviceType () const {
            return *_device;
         }

         BaseThread & startThread ( WorkDescriptor &wd, SchedulingGroup *sg = 0 );
         virtual BaseThread & createThread ( WorkDescriptor &wd ) = 0;
         BaseThread & associateThisThread ( SchedulingGroup *sg, bool untieMain=true );

         BaseThread & startWorker ( SchedulingGroup *sg );
         void stopAll();

         /* capabilitiy query functions */
         virtual bool supportsUserLevelThreads() const = 0;
         virtual bool hasSeparatedMemorySpace() const = 0;

         /* Memory space suport */
         virtual void copyDataIn( WorkDescriptor& wd ) {}
         virtual void copyDataOut( WorkDescriptor& wd ) {}

         virtual void* getAddress( WorkDescriptor& wd, uint64_t tag, nanos_sharing_t sharing );
         virtual void copyTo( WorkDescriptor& wd, void *dst, uint64_t tag, nanos_sharing_t sharing, size_t size );
   };

   typedef class ProcessingElement PE;
   typedef PE * ( *peFactory ) ( int pid );
};

#endif
