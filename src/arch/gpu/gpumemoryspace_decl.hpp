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

#ifndef GPUMEMORYSPACE_DECL_H
#define GPUMEMORYSPACE_DECL_H

#include "simpleallocator_fwd.hpp"
#include "gpuprocessor_decl.hpp"

namespace nanos {
namespace ext {

   class GPUMemorySpace {
      bool             _initialized;
      SimpleAllocator *_allocator;
      std::size_t      _memoryAlignment;
      BufferManager    _inputPinnedMemoryBuffer;
      BufferManager    _outputPinnedMemoryBuffer;
      GPUProcessor    *_gpu;

      public:
      GPUMemorySpace();
      void initialize( bool allocateInputMem, bool allocateOutputMem, GPUProcessor *gpu );
      SimpleAllocator *getAllocator() const;
      std::size_t getAlignment() const;
      void freeOutputPinnedMemory ();
      void * allocateOutputPinnedMemory ( std::size_t size );
      void freeInputPinnedMemory ();
      void * allocateInputPinnedMemory ( std::size_t size );
      GPUProcessor * getGPU() const;
   };
} // namespace ext
} // namespace nanos

#endif /* GPUMEMORYSPACE_DECL_H */
