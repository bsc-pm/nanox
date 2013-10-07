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

#ifndef _NANOS_ARCHPLUGIN_DECL
#define _NANOS_ARCHPLUGIN_DECL

#include "plugin_decl.hpp"
#include "processingelement_fwd.hpp"
#include <vector>

namespace nanos
{

   /** \brief Base class for specific architecture plugins
    */
   class ArchPlugin : public Plugin
   {

      private:
         //! ProcessingElement list
         typedef std::vector<unsigned> PEList;
         
         //! PE Binding list for this arch
         PEList _bindings;
         
      public:
         /** \brief Constructs the plugin and registers itself in System. */
         ArchPlugin( const char *name, int version );
         
         /** \brief Adds a CPU id to a list to bind this arch to it later.
          */
         void addBinding ( unsigned cpu_id );

         /** \brief Returns the cpu id the given PE should be binded to.
          *  \param pe The index processing element that you want to create.
          *  This index is relative to the architecture. That means it takes
          *  a value between 0 and getNumPEs() - 1.
          */
         PEList::value_type getBinding( unsigned index );
         
         /** \brief Returns the number of PEs that this plugin will try to
          * create in a given NUMA node.
          *  \param node Node number.
          */
         //virtual unsigned getPEsInNode( unsigned node ) const = 0;
         
         /** \brief Returns the number of helper PEs this plugin requires.
          * This number is added to the number of SMP PEs.
          * For instance, the synchronous version of the CUDA GPU plugin
          * requires a dedicated hardware thread for each GPU. The async version
          * should not required any, and the SMP plugin should also return 0.
          */
         virtual unsigned getNumHelperPEs() const;

         /** \brief Number of PEs to be used by this architecture.
          */
         virtual unsigned getNumPEs() const = 0;
        
         /** \brief Returns the maximum number of threads.
          *  This will be used to compute System::_targetThreads.
          */
         virtual unsigned getNumThreads() const = 0; 
         
         /** \brief Instructs the plugin to fill the PE binding list.
          * Plugins which need helper PEs (e.g. CUDA GPU) will use their own binding list
          * instead the one available in System.
          */
         virtual void createBindingList();

         /** \brief Creates a PE.
          *  \param id Number of the PE of this architecture.
          *  For instance, in CUDA GPU, this value cannot be higher than
          *  the number of GPUs used.
          *  \param uid Unique ID of this PE. The uniqueness of this number
          *  must be guaranteed by the caller.
          *  \note Be sure to call this as many times as what
          *  getNumPEs() returns.
          *  \return A pointer to a PE to be added by System to
          *  the PEs vector.
          */
         virtual ProcessingElement * createPE( unsigned id, unsigned uid ) = 0;
   };
}

#endif
