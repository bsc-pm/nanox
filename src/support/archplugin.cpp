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

#include "archplugin.hpp"
#include "system.hpp"

using namespace nanos;

ArchPlugin::ArchPlugin( const char *name, int version ) : Plugin( name, version )
{
   sys.registerArchitecture( this );
}

unsigned ArchPlugin::getNumHelperPEs() const
{
   return 0;
}

void ArchPlugin::createBindingList()
{
}

void ArchPlugin::initialize() {
   std::cerr << "Generic " << __FUNCTION__ << std::endl;
}
void ArchPlugin::finalize() {
   std::cerr << "Generic " << __FUNCTION__ << std::endl;
}
void ArchPlugin::addPEs( PEMap &pes ) const {
   std::cerr << "Generic " << __FUNCTION__ << std::endl;
}
void ArchPlugin::addDevices( DeviceList &devices ) const {
   std::cerr << "Generic " << __FUNCTION__ << std::endl;
}
void ArchPlugin::startSupportThreads() {
   std::cerr << "Generic " << __FUNCTION__ << std::endl;
}
void ArchPlugin::startWorkerThreads( std::map<unsigned int, BaseThread *> &workers ) {
   std::cerr << "Generic " << __FUNCTION__ << std::endl;
}
unsigned int ArchPlugin::getMaxPEs() const {
   std::cerr << "Generic " << __FUNCTION__ << std::endl;
   return 0;
}
unsigned int ArchPlugin::getNumWorkers() const {
   std::cerr << "Generic " << __FUNCTION__ << std::endl;
   return 0;
}
unsigned int ArchPlugin::getMaxWorkers() const {
   std::cerr << "Generic " << __FUNCTION__ << std::endl;
   return 0;
}
