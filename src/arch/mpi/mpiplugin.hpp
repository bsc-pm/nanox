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

#ifndef _NANOS_MPIPLUGIN_HPP
#define _NANOS_MPIPLUGIN_HPP

#include "atomic.hpp"
#include "plugin.hpp"

namespace nanos {
namespace ext {

class MPIPlugin : public ArchPlugin {
   //The three boleans below implement the initialization order
   //First system will "pre-initialize" before initializing threads (not if extrae enabled and only when we are slaves)
   //Then extrae will initialize
   //Then system will "post-initialice" as part of user main (ompss_nanox_main)
   bool _extraeInitialized;
   bool _initialized;
   bool _preinitialized;
   static Atomic<unsigned int> _numWorkers;
   static Atomic<unsigned int> _numPEs;
   
   public:
    MPIPlugin() : ArchPlugin( "MPI PE Plugin",1 ), _extraeInitialized(false),_initialized(false), _preinitialized(false) {}

    virtual void config ( Config& cfg );

    virtual bool configurable();

    virtual void init();

    static void addPECount(unsigned int count);

    static void addWorkerCount(unsigned int count);

    virtual unsigned getNumThreads() const;

    virtual unsigned int getNumPEs() const;

    virtual unsigned int getMaxPEs() const;

    virtual unsigned int getNumWorkers() const;

    virtual unsigned int getMaxWorkers() const;

    virtual void createBindingList();
       
    virtual void addPEs( PEMap &pes ) const;

    virtual void addDevices( DeviceList &devices ) const;

    virtual void startSupportThreads();

    virtual void startWorkerThreads( std::map<unsigned int, BaseThread *> &workers );
    
    virtual void finalize();

    virtual PE* createPE( unsigned id, unsigned uid );
};

} // namespace ext
} // namespace nanos

#endif // _NANOS_MPIPLUGIN_HPP

