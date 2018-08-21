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

#ifndef _NANOS_SMPBASEPLUGIN_DECL
#define _NANOS_SMPBASEPLUGIN_DECL

#include <fstream>
#include "cpuset.hpp"
#include "smpprocessor_fwd.hpp"
#include "smpthread_fwd.hpp"
#include "threadteam_fwd.hpp"
#include "archplugin_decl.hpp"

namespace nanos {

class SMPBasePlugin : public ArchPlugin {
   public:
      SMPBasePlugin( const char *name, int version ) : ArchPlugin( name, version ) {}
      virtual ext::SMPProcessor *getFirstSMPProcessor() const = 0;
      virtual ext::SMPProcessor *getLastFreeSMPProcessorAndReserve() = 0;
      virtual ext::SMPProcessor *getLastSMPProcessor() = 0;
      virtual ext::SMPProcessor *getFreeSMPProcessorByNUMAnodeAndReserve(int node) = 0;
      virtual ext::SMPProcessor *getSMPProcessorByNUMAnode(int node, unsigned int idx) const = 0;
      virtual bool getBinding() const = 0;
      virtual int getCpuCount() const = 0;
      virtual void admitCurrentThread( std::map<unsigned int, BaseThread *> &workers, bool isWorker ) = 0;
      virtual void expelCurrentThread( std::map<unsigned int, BaseThread *> &workers, bool isWorker ) = 0;
      virtual int getNumSockets() const = 0;
      virtual int getCurrentSocket() const = 0;
      virtual void setCurrentSocket( int socket ) = 0;
      virtual void setNumSockets ( int numSockets ) = 0;
      virtual void setCPUsPerSocket ( int cpus_per_socket ) = 0;
      virtual int getCPUsPerSocket() const = 0;
      virtual unsigned int getNewSMPThreadId() = 0;
      virtual void updateActiveWorkers ( int nthreads, std::map<unsigned int, BaseThread *> &workers, ThreadTeam *team ) = 0;
      virtual void updateCpuStatus( int cpuid ) = 0;
      virtual const CpuSet& getCpuProcessMask() const = 0 ;
      virtual bool setCpuProcessMask( const CpuSet& mask, std::map<unsigned int, BaseThread *> &workers ) = 0;
      virtual void addCpuProcessMask( const CpuSet& mask, std::map<unsigned int, BaseThread *> &workers ) = 0;
      virtual const CpuSet& getCpuActiveMask() const = 0 ;
      virtual bool setCpuActiveMask( const CpuSet& mask, std::map<unsigned int, BaseThread *> &workers ) = 0;
      virtual void addCpuActiveMask( const CpuSet& mask, std::map<unsigned int, BaseThread *> &workers ) = 0;
      virtual void enableCpu( int cpuid, std::map<unsigned int, BaseThread *> &workers ) = 0;
      virtual void disableCpu( int cpuid, std::map<unsigned int, BaseThread *> &workers ) = 0;
      virtual void forceMaxThreadCreation( std::map<unsigned int, BaseThread *> &workers ) = 0;
      virtual ext::SMPThread &associateThisThread( bool untie ) = 0;
      virtual void setRequestedWorkers( int workers ) = 0;
      virtual int getRequestedWorkers() const = 0;
      virtual unsigned int getMaxWorkers() const = 0;
      virtual void createWorker( std::map<unsigned int, BaseThread *> &workers ) = 0;
      virtual std::pair<std::string, std::string> getBindingStrings() const = 0;
      virtual bool asyncTransfersEnabled() const = 0;
};

} // namespace nanos

#endif /* _NANOS_SMPBASEPLUGIN_DECL */
