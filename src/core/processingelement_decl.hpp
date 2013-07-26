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

#ifndef _NANOS_PROCESSING_ELEMENT_DECL
#define _NANOS_PROCESSING_ELEMENT_DECL

#include "workdescriptor_decl.hpp"
#include <algorithm>
#include "functors_decl.hpp"
#include "basethread_fwd.hpp"
#include "schedule_fwd.hpp"

namespace nanos
{
   class ProcessingElement
   {
      private:
         typedef std::vector<BaseThread *>    ThreadList;
         int                                  _id;
         //! Unique ID
         int                                  _uid;
         const Device *                       _device;
         ThreadList                           _threads;
         int                                  _numaNode;

      private:
         /*! \brief ProcessinElement default constructor
          */
         ProcessingElement ();
         /*! \brief ProcessinElement copy constructor (private)
          */
         ProcessingElement ( const ProcessingElement &pe );
         /*! \brief ProcessinElement copy assignment operator (private)
          */
         const ProcessingElement & operator= ( const ProcessingElement &pe );
      protected:
         virtual WorkDescriptor & getMasterWD () const = 0;
         virtual WorkDescriptor & getWorkerWD () const = 0;
      public:
         /*! \brief ProcessinElement constructor
          */
         ProcessingElement ( int newId, const Device *arch, int uniqueId ) : _id ( newId ), _uid( uniqueId ), _device ( arch ), _numaNode( 0 ) {}

         /*! \brief ProcessinElement destructor
          */
         virtual ~ProcessingElement();

         /* get/put methods */
         int getId() const;
         
         //! \brief Returns a unique ID that no other PE will have.
         int getUId() const;
         
         //! \brief Returns the socket this thread is running on.
         int getNUMANode() const;
         
         //! \brief Sets the socket this thread is running on.
         void setNUMANode( int node );

         const Device & getDeviceType () const;

         BaseThread & startThread ( WorkDescriptor &wd );
         virtual BaseThread & createThread ( WorkDescriptor &wd ) = 0;
         BaseThread & associateThisThread ( bool untieMain=true );

         BaseThread & startWorker ( );
         void stopAll();

         /* capabilitiy query functions */
         virtual bool supportsUserLevelThreads() const = 0;
         virtual bool hasSeparatedMemorySpace() const { return false; }
         virtual unsigned int getMemorySpaceId() const { return 0; }

         virtual void waitInputDependent( uint64_t tag ) {}

         /* Memory space suport */
         virtual void copyDataIn( WorkDescriptor& wd );
         virtual void copyDataOut( WorkDescriptor& wd );

         virtual void waitInputs( WorkDescriptor& wd );

         virtual void* getAddress( WorkDescriptor& wd, uint64_t tag, nanos_sharing_t sharing );
         virtual void copyTo( WorkDescriptor& wd, void *dst, uint64_t tag, nanos_sharing_t sharing, size_t size );

         /*!
          * \brief Returns the first thread of the PE that has team and is not tagged to sleep
          */
         virtual BaseThread* getFirstRunningThread();

         /*!
          * \brief Returns the first thread of the PE that has no team or is tagged to sleep
          */
         virtual BaseThread* getFirstStoppedThread();
   };

   typedef class ProcessingElement PE;
   typedef PE * ( *peFactory ) ( int pid, int uid );
};

#endif
