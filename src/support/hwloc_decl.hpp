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

#ifndef HWLOC_DECL
#define HWLOC_DECL


#include <config.hpp>
#include <string>
#include "cpuset.hpp"

#ifdef HWLOC
#include <hwloc.h>
#endif

namespace nanos {

class Hwloc {
#ifdef HWLOC
   //! hwloc topology structure
   hwloc_topology_t             _hwlocTopology;
   //! Path to a hwloc topology xml
   std::string                  _topologyPath;
#endif

   public:

      Hwloc();
      ~Hwloc();
      
      void config( Config & cfg );
      bool isHwlocAvailable() const;
   
      void loadHwloc();
      void unloadHwloc();
      unsigned int getNumaNodeOfCpu( unsigned int cpu );
      unsigned int getNumaNodeOfGpu( unsigned int gpu );
      void getNumSockets(unsigned int &allowedNodes, int &numSockets, unsigned int &hwThreads);

      /*!
       * \brief Checks if we can see the CPU, to create the PE.
       * If hwloc has no info on that CPU, we should not continue creating
       * a PE for it, since it will hardly be available.
       *
       * If hwloc is not available, this function returns true.
       *
       * @param cpu OS CPU index.
       */
      bool isCpuAvailable( unsigned int cpu ) const;

      CpuSet getCoreCpusetOf( unsigned int cpu );
      std::list<CpuSet> getCoreCpusetsOf( const CpuSet& parent );
};

} // namespace nanos

#endif /* HWLOC_DECL */
