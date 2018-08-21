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

#ifndef NANOS_PM_INTERFACE_DECL
#define NANOS_PM_INTERFACE_DECL

#include "config_fwd.hpp"
#include "workdescriptor_decl.hpp"
#include "threadteam_decl.hpp"
#include "cpuset.hpp"

namespace nanos {

   class PMInterface
   {
      protected:
         std::string    _description; /*!< \brief String describing Programming Model Interface */
         bool           _malleable;   /*!< \brief Can the PM dynamically change the team size? */
      private:
         /*! \brief PMInterface copy constructor (private)
          */
         PMInterface( PMInterface &pmi );
         /*! \brief PMInterface copy assignment operator (private)
          */
         PMInterface& operator= ( PMInterface &pmi );
      public:
         enum Interfaces{ OmpSs, OpenMP, None }; 
      public:
         /*! \brief PMInterface default constructor
          */
         PMInterface() : _description(), _malleable(true) {}
         /*! \brief PMInterface destructor
          */
         virtual ~PMInterface() {}

         virtual int getInternalDataSize() const { return 0; }
         virtual int getInternalDataAlignment() const { return 1; }
         virtual void initInternalData( void *data ) {}

         virtual void config (nanos::Config &cfg) {}
         virtual void start () { _description = std::string("none"); }
         virtual void finish() {}

         virtual void setupWD( nanos::WD &wd ) {}
         virtual void wdStarted( nanos::WD &wd ) {}
         virtual void wdFinished( nanos::WD &wd ) {}

         virtual nanos::ThreadTeamData* getThreadTeamData() { return NEW nanos::ThreadTeamData(); }
         std::string getDescription( void ) { return _description; }

         bool isMalleable( void ) const { return _malleable; }
         bool isOmpSs( void ) const { return _malleable; }

         virtual int getMaxThreads() const { return 0; }
         virtual void setNumThreads( int nthreads ) {}
         virtual void setNumThreads_globalState( int nthreads ) {}

         virtual const CpuSet& getCpuProcessMask() const = 0;
         virtual bool setCpuProcessMask( const CpuSet& cpu_set ) { return false; }
         virtual void addCpuProcessMask( const CpuSet& cpu_set ) {}
         virtual const CpuSet& getCpuActiveMask() const = 0;
         virtual bool setCpuActiveMask( const CpuSet& cpu_set ) { return false; }
         virtual void addCpuActiveMask( const CpuSet& cpu_set ) {}
         virtual void enableCpu( int cpuid ) {}
         virtual void disableCpu( int cpuid ) {}
#ifdef DLB
         virtual void registerCallbacks() const {}
#endif

         //! By default, OmpSs is assumed (required for the bare run in system.cpp)
         virtual Interfaces getInterface() const { return PMInterface::OmpSs; }
   };

} // namespace nanos

#endif /* PM_INTERFACE_HPP_ */
