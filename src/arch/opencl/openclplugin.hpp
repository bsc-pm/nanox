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

//#include "openclconfig.hpp"
//#include "os.hpp"
//#include "processingelement_fwd.hpp"
//#include "plugin.hpp"
#include "archplugin.hpp"
#include "openclprocessor.hpp"
#include "openclthread_decl.hpp"
//#include "smpprocessor.hpp"

#ifdef HAVE_OPENCL_OPENCL_H
#include <OpenCL/opencl.h>
#endif

#ifdef HAVE_CL_OPENCL_H
#include <CL/opencl.h>
#endif

namespace nanos {
namespace ext {

class OpenCLPlugin : public ArchPlugin
{
   
private:
    
   static std::string _devTy;
  // All found devices.
   static std::map<cl_device_id, cl_context> _devices;
   std::vector<ext::OpenCLProcessor *> _opencls;
   std::vector<OpenCLThread *>  _openclThreads;

   friend class OpenCLConfig;
   
public:
   OpenCLPlugin() : ArchPlugin( "OpenCL PE Plugin", 1 )
      , _opencls()
      , _openclThreads()
   { }

   ~OpenCLPlugin() { }

   void config( Config &cfg );
   void init();
   //virtual unsigned getPEsInNode( unsigned node ) const;
   virtual unsigned getNumHelperPEs() const;
   //virtual unsigned getNumPEs() const;
   virtual unsigned getNumThreads() const;
   virtual void createBindingList();
   virtual PE* createPE( unsigned id, unsigned uid );
   virtual void addPEs( PEMap &pes ) const;
   virtual void addDevices( DeviceList &devices ) const;
   virtual void startSupportThreads();
   virtual void startWorkerThreads( std::map<unsigned int, BaseThread *> &workers );
   virtual void finalize();

   virtual std::string getSelectedDeviceType() const {
      return _devTy;
   }

   virtual unsigned int getNumPEs() const {
      return _opencls.size();
   }
   
   virtual unsigned int getMaxPEs() const {
      return _opencls.size();
   }
   
   virtual unsigned int getNumWorkers() const {
      return _opencls.size();
   }
   
   virtual unsigned int getMaxWorkers() const {
      return _opencls.size();
   }

}; // class OpenCLPlugin

} // namespace ext
} // namespace nanos

