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

#include "nanos-int.h"
#include "workdescriptor_fwd.hpp"
#include "remoteworkdescriptor_decl.hpp"

namespace nanos {
namespace ext {

   class SerializedWDFields {
      unsigned int _wdId;
      int          _archId;
      void       (*_outline)(void *);
      void       (*_xlate)(void *, void*);
      std::size_t  _dataSize;
      std::size_t  _numCopies;
      std::size_t  _totalDimensions;
      const char  *_descriptionAddr;
      WD const    *_wd;
      public:
      static std::size_t getTotalSize( WD const &wd );
      void setup( WD const &wd );
      CopyData *getCopiesAddr() const;
      nanos_region_dimension_internal_t *getDimensionsAddr() const;
      char *getDataAddr() const;
      std::size_t getTotalDimensions() const;
      std::size_t getDataSize() const;
      void (*getXlateFunc() const)(void *, void*);
      void (*getOutline() const)(void *);
      unsigned int getArchId() const;
      unsigned int getWDId() const;
      WD const *getWDAddr() const;
      unsigned int getNumCopies() const;
      const char *getDescriptionAddr() const;
   };

   class WD2Net {
      std::size_t _bufferSize;
      std::size_t _dataSize;
      char *_buffer;
      void build();
      WD2Net( WD2Net const &nwd );
      WD2Net &operator=( WD2Net const & );
      public:
      WD2Net( WD const &wd );
      ~WD2Net();
      char *getBuffer() const;
      std::size_t getBufferSize() const;
   };

   class Net2WD {
      WD *_wd;
      public:
      Net2WD( char *buffer, std::size_t buffer_size, RemoteWorkDescriptor **rwds );
      ~Net2WD();
      WD *getWD();
   };
} // namespace ext
} // namespace nanos

