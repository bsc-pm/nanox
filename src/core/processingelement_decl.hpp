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
   namespace ext {
   class SMPMultiThread;
   };

   class ProcessingElement
   {
      private:
         typedef std::vector<BaseThread *>    ThreadList;
         int                                  _id;
         const Device *                       _device;
         const Device *                       _subDevice;
         const Device *                       _deviceNo;
         const Device *                       _subDeviceNo;
         ThreadList                           _threads;
         unsigned int                         _memorySpaceId;
         int                                  _numaNode;

      private:
         /*! \brief ProcessingElement default constructor
          */
         ProcessingElement ();
         /*! \brief ProcessingElement copy constructor (private)
          */
         ProcessingElement ( const ProcessingElement &pe );
         /*! \brief ProcessingElement copy assignment operator (private)
          */
         const ProcessingElement & operator= ( const ProcessingElement &pe );
      protected:
         virtual WorkDescriptor & getMasterWD () const = 0;
         virtual WorkDescriptor & getWorkerWD () const = 0;
         virtual WorkDescriptor & getMultiWorkerWD () const = 0;
      public:
         /*! \brief ProcessingElement constructor
          */
         ProcessingElement ( int newId, const Device *arch, const Device *subArch, unsigned int memSpaceId ) : _id ( newId ), _device ( arch ), _subDevice( subArch ), _deviceNo ( NULL ), _subDeviceNo ( NULL ), _memorySpaceId( memSpaceId ), _numaNode( 0 ) {}

         /*! \brief ProcessingElement destructor
          */
         virtual ~ProcessingElement();

         /* get/put methods */
         int getId() const;
         
         //! \brief Returns the socket this thread is running on.
         int getNUMANode() const;
         
         //! \brief Sets the socket this thread is running on.
         void setNUMANode( int node );

         const Device * getDeviceType () const;
         const Device * getSubDeviceType () const;
         virtual const Device * getCacheDeviceType () const;

         BaseThread & startThread ( WorkDescriptor &wd, ext::SMPMultiThread *parent=NULL );
         BaseThread & startMultiThread ( WorkDescriptor &wd, unsigned int numPEs, ProcessingElement **repPEs );
         virtual BaseThread & createThread ( WorkDescriptor &wd, ext::SMPMultiThread *parent=NULL ) = 0;
         virtual BaseThread & createMultiThread ( WorkDescriptor &wd, unsigned int numPEs, ProcessingElement **repPEs ) = 0;

         BaseThread & associateThisThread ( bool untieMain=true );

         BaseThread & startWorker ( ext::SMPMultiThread *parent=NULL );
         BaseThread & startMultiWorker ( unsigned int numPEs, ProcessingElement **repPEs );
         
         void disableDevice( int num )
{
 if ( num == 0 ) { _deviceNo = _device; _device = NULL; } 
 if ( num == 1 ) { _subDeviceNo = _subDevice; _subDevice = NULL; } 
}
         void enableDevice( int num )
{
 if ( num == 0 ) { _device = _deviceNo; _deviceNo = NULL; } 
 if ( num == 1 ) { _subDevice = _subDeviceNo; _subDeviceNo = NULL; } 
}

         void stopAll();

         /* capabilitiy query functions */
         virtual bool supportsUserLevelThreads() const = 0;
         virtual bool hasSeparatedMemorySpace() const { return false; }
         unsigned int getMemorySpaceId() const { return _memorySpaceId; }
         virtual unsigned int getMyNodeNumber() const { return 0; }

         /* Memory space suport */
         void copyDataIn( WorkDescriptor& wd );
         virtual void copyDataOut( WorkDescriptor& wd );
         //virtual bool dataCanBlockUs( WorkDescriptor& wd );

         virtual void waitInputs( WorkDescriptor& wd );
         bool testInputs( WorkDescriptor& wd );

         virtual bool isGPU() const = 0;
         BaseThread *getFirstThread() const { return _threads[0]; }

         virtual BaseThread* getFirstRunningThread();
         virtual BaseThread* getFirstStoppedThread();
   };

   typedef class ProcessingElement PE;
   typedef PE * ( *peFactory ) ( int pid );
};

#endif
