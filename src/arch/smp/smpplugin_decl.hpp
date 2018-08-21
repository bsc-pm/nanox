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

#ifndef _NANOS_SMP_PLUGIN
#define _NANOS_SMP_PLUGIN

#include <iostream>

#include "atomic.hpp"
#include "debug.hpp"
#include "smpbaseplugin_decl.hpp"
#include "plugin.hpp"
#include "smpprocessor.hpp"
#include "os.hpp"

#include "cpuset.hpp"
#include <limits>

#ifdef HAVE_MEMKIND_H
#include <memkind.h>
#endif

namespace nanos {
namespace ext {

nanos::PE * smpProcessorFactory ( int id, int uid );


class SMPPlugin : public SMPBasePlugin
{
   protected:
   //! CPU id binding list
   typedef std::vector<int> Bindings;

   Atomic<unsigned int>         _idSeed;
   int                          _requestedCPUs;
   int                          _availableCPUs;
   int                          _currentCPUs;
   int                          _requestedWorkers;
   std::vector<SMPProcessor *> *_cpus;
   std::map<int,SMPProcessor*> *_cpusByCpuId;
   std::vector<SMPThread *>     _workers;
   int                          _bindingStart;
   int                          _bindingStride;
   bool                         _bindThreads;
   bool                         _smpPrivateMemory;
   bool                         _smpAllocWide;
   int                          _smpHostCpus;
   std::size_t                  _smpPrivateMemorySize;
   bool                         _workersCreated;
   int                          _threadsPerCore;

   // Nanos++ scheduling domain
   CpuSet                       _cpuSystemMask;   /*!< \brief system's default cpu_set */
   CpuSet                       _cpuProcessMask;  /*!< \brief process' default cpu_set */
   CpuSet                       _cpuActiveMask;   /*!< \brief mask of current active cpus */

   //! Physical NUMA nodes
   int                          _numSockets;
   int                          _CPUsPerSocket;
   //! The socket that will be assigned to the next WD
   int                          _currentSocket;


   //! CPU id binding list
   Bindings                     _bindings;

   bool                         _memkindSupport;
   std::size_t                  _memkindMemorySize;
   bool                         _asyncSMPTransfers;

   public:
   SMPPlugin();

   ~SMPPlugin();

   virtual unsigned int getNewSMPThreadId();

   virtual void config ( Config& cfg );

   virtual void init();

   virtual unsigned int getEstimatedNumThreads() const;

   virtual unsigned int getNumThreads() const;

   virtual ProcessingElement* createPE( unsigned id, unsigned uid );

   virtual void initialize();

   virtual void finalize();

   virtual void addPEs( PEMap &pes ) const;

   virtual void addDevices( DeviceList &devices ) const;

   virtual void startSupportThreads();

   virtual void startWorkerThreads( std::map<unsigned int, BaseThread *> &workers );

   virtual void setRequestedWorkers( int workers );

   virtual int getRequestedWorkers( void ) const;

   virtual ext::SMPProcessor *getFirstSMPProcessor() const;

   virtual ext::SMPProcessor *getFirstFreeSMPProcessor() const;

   virtual ext::SMPProcessor *getLastFreeSMPProcessorAndReserve();

   virtual ext::SMPProcessor *getLastSMPProcessor();

   virtual ext::SMPProcessor *getFreeSMPProcessorByNUMAnodeAndReserve(int node);

   virtual ext::SMPProcessor *getSMPProcessorByNUMAnode(int node, unsigned int idx) const;

   void loadNUMAInfo ();

   unsigned getNodeOfPE ( unsigned pe );

   void setBindingStart ( int value );

   int getBindingStart () const;

   void setBindingStride ( int value );

   int getBindingStride () const;

   void setBinding ( bool set );

   virtual bool getBinding () const;

   virtual int getNumSockets() const;

   virtual void setNumSockets ( int numSockets );

   virtual int getCurrentSocket() const;

   virtual void setCurrentSocket( int currentSocket );

   virtual int getCPUsPerSocket() const;

   virtual void setCPUsPerSocket ( int cpus_per_socket );

   virtual const CpuSet& getCpuProcessMask () const;

   virtual bool setCpuProcessMask ( const CpuSet& mask, std::map<unsigned int, BaseThread *> &workers );

   virtual void addCpuProcessMask ( const CpuSet& mask, std::map<unsigned int, BaseThread *> &workers );

   virtual const CpuSet& getCpuActiveMask () const;

   virtual bool setCpuActiveMask ( const CpuSet& mask, std::map<unsigned int, BaseThread *> &workers );
   
   virtual void addCpuActiveMask ( const CpuSet& mask, std::map<unsigned int, BaseThread *> &workers );

   virtual void enableCpu ( int cpuid, std::map<unsigned int, BaseThread *> &workers );

   virtual void disableCpu ( int cpuid, std::map<unsigned int, BaseThread *> &workers );

   virtual void updateActiveWorkers ( int nthreads, std::map<unsigned int, BaseThread *> &workers, ThreadTeam *team );

   virtual void updateCpuStatus( int cpuid );

   virtual void admitCurrentThread( std::map<unsigned int, BaseThread *> &workers, bool isWorker );

   virtual void expelCurrentThread( std::map<unsigned int, BaseThread *> &workers, bool isWorker );

   virtual int getCpuCount() const;

   virtual unsigned int getNumPEs() const;

   virtual unsigned int getMaxPEs() const;

   unsigned int getEstimatedNumWorkers() const;

   virtual unsigned int getNumWorkers() const;

   //! \brief Get the max number of Workers that could run with the current Active Mask
   virtual unsigned int getMaxWorkers() const;

   virtual SMPThread &associateThisThread( bool untie );

   /*! \brief Force the creation of at least 1 thread per CPU.
    */
   virtual void forceMaxThreadCreation( std::map<unsigned int, BaseThread *> &workers );

       /*! \brief Create a worker in a suitable CPU
    */
   void createWorker( std::map<unsigned int, BaseThread *> &workers );

   virtual std::pair<std::string, std::string> getBindingStrings() const;

protected:

   void applyCpuMask ( std::map<unsigned int, BaseThread *> &workers );

   void createWorker( ext::SMPProcessor *target, std::map<unsigned int, BaseThread *> &workers );

   bool isValidMask( const CpuSet& mask ) const;

   virtual bool asyncTransfersEnabled() const;
};
}
}

#endif  // _NANOS_SMP_PLUGIN
