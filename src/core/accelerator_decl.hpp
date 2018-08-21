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

#ifndef _NANOS_ACCELERATOR_DECL
#define _NANOS_ACCELERATOR_DECL

#include <stdint.h>
#include "workdescriptor_decl.hpp"
#include "processingelement_decl.hpp"
#include "functors_decl.hpp"
#include "atomic_decl.hpp"
#include "copydescriptor_decl.hpp"

namespace nanos {

//   class Accelerator : public ProcessingElement
//   {
//      protected:
//         virtual WorkDescriptor & getMasterWD () const = 0;
//         virtual WorkDescriptor & getWorkerWD () const = 0;
//
//      private:
//        /*! \brief Accelerator default constructor (private)
//         */
//         Accelerator ();
//        /*! \brief Accelerator copy constructor (private)
//         */
//         Accelerator ( const Accelerator &a );
//        /*! \brief Accelerator copy assignment operator (private)
//         */
//         const Accelerator& operator= ( const Accelerator &a );
//      public:
//        /*! \brief Accelerator constructor - from 'newId' and 'arch'
//         */
//         Accelerator ( const Device *arch, const Device *subArch, memory_space_id_t memId );
//        /*! \brief Accelerator destructor
//         */
//         virtual ~Accelerator() {}
//
//         virtual bool hasSeparatedMemorySpace() const { return true; }
//
//         virtual void copyDataOut( WorkDescriptor& wd );
//
//         virtual void waitInputs( WorkDescriptor& wd );
//
//         virtual void waitInputsDependent( WorkDescriptor &wd ) = 0;
//   };

} // namespace nanos

#endif
