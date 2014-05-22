/*************************************************************************************/
/*      Copyright 2013 Barcelona Supercomputing Center                               */
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

#ifndef _NANOS_SMPBASEPLUGIN_DECL
#define _NANOS_SMPBASEPLUGIN_DECL

#include <sched.h>
#include "smpprocessor_fwd.hpp"
#include "smpthread_fwd.hpp"
#include "archplugin_decl.hpp"

namespace nanos {

class SMPBasePlugin : public ArchPlugin {
   public:
      SMPBasePlugin( const char *name, int version ) : ArchPlugin( name, version ) {}
      virtual ext::SMPProcessor *getFirstSMPProcessor() const = 0;
      virtual cpu_set_t &getActiveSet() = 0;
      virtual ext::SMPProcessor *getLastFreeSMPProcessorAndReserve() = 0;
      virtual ext::SMPProcessor *getFreeSMPProcessorByNUMAnodeAndReserve(int node) = 0;
      virtual ext::SMPProcessor *getSMPProcessorByNUMAnode(int node, unsigned int idx) const = 0;
      virtual bool getBinding() const = 0;
      virtual int getCpuCount() const = 0;
      virtual void admitCurrentThread( std::vector<BaseThread *> &workers ) = 0;
      virtual int getNumSockets() const = 0;
      virtual int getCurrentSocket() const = 0;
      virtual void setCurrentSocket( int socket ) = 0;
      virtual int getNumAvailSockets() const = 0;
      virtual int getVirtualNUMANode( int physicalNode ) const = 0;
      virtual void setNumSockets ( int numSockets ) = 0;
      virtual void setCoresPerSocket ( int coresPerSocket ) = 0;
      virtual int getCoresPerSocket() const = 0;
      virtual unsigned int getNewSMPThreadId() = 0;
      virtual void updateActiveWorkers ( int nthreads ) = 0;
      virtual void setCpuMask ( const cpu_set_t *mask ) = 0;
      virtual void getCpuMask ( cpu_set_t *mask ) const = 0;
      virtual void addCpuMask ( const cpu_set_t *mask ) = 0;
      virtual ext::SMPThread &associateThisThread( bool untie ) = 0;
      virtual void setRequestedWorkers( int workers ) = 0;
      virtual int getRequestedWorkersOMPSS() const = 0;
};

}

#endif /* _NANOS_SMPBASEPLUGIN_DECL */
