/*************************************************************************************/
/*      Copyright 2009 Barcelona Supercomputing Center                               */
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

#ifndef _NANOS_OCL_WD
#define _NANOS_OCL_WD

#include "debug.hpp"
#include "ocldevice_decl.hpp"
#include "workdescriptor.hpp"

namespace nanos {
namespace ext {

extern OCLDevice OCLDev;

// OpenCL back-end Device Description.
//
// Since the OpenCL back-end is slightly different from a normal back-end, we
// have a set of specialized Device Description. This is the root of the class
// hierarchy.
//
// Every Device Description must have the following layout in memory:
//
// +------------------+
// | Custom section   |
// +------------------+
// | Number of events |
// +------------------+
// | Input event 1    |
// | ...              |
// | Input event n    |
// +------------------+
// | Output event     |
// +------------------+ -+-
// | Start tick       |  | Profiling section
// | End tick         |  | (optional)
// +------------------+ -+-
//
// The OLCDD class contains the EventIterator class that can be use to iterate
// over events given the custom section size. It is Device Description builder
// responsibility to push arguments in the right order!

class OCLDD : public DD
{
public:
  struct EvData
  {
     unsigned _evCount;
  } __attribute__((packed));

  struct ProfData
  {
     unsigned long long *_startTick;
     unsigned long long *_endTick;
  } __attribute__((packed));

  class EventIterator
  {
  public:
     static EventIterator begin( void *args, size_t offset = 0 )
     {
        uint8_t *metaBase = static_cast<uint8_t *>( args ) + offset;
        EvData *meta = reinterpret_cast<EvData *>( metaBase );

        uint8_t *evBase = reinterpret_cast<uint8_t *>( meta ) +
                          sizeof( EvData );

        return EventIterator( reinterpret_cast<int **>( evBase ) );
     }

     static EventIterator end( void *args, size_t offset = 0 )
     {
        uint8_t *metaBase = static_cast<uint8_t *>( args ) + offset;
        EvData *meta = reinterpret_cast<EvData *>( metaBase );

        uint8_t *evBase = reinterpret_cast<uint8_t *>( meta ) +
                          sizeof( EvData );

        return EventIterator( reinterpret_cast<int **>( evBase ) +
                              meta->_evCount );
     }

  private:
     EventIterator( int **evs ) : _cur( evs ) { }

  public:
     bool operator==(const EventIterator &iter) const
     {
        return _cur == iter._cur;
     }

     bool operator!=(const EventIterator &iter) const
     {
        return _cur != iter._cur;
     }

     int *&operator*() const { return *_cur; }

     EventIterator &operator++()
     {
        ++_cur; return *this;
     }

     EventIterator operator++(int ign)
     {
        EventIterator iter = *this; ++(*this); return iter;
     }

     EventIterator operator-( int offset ) const
     {
        return EventIterator( _cur - offset );
     }

     EventIterator operator+( int offset ) const
     {
        return EventIterator( _cur + offset );
     }

  public:
     void **getRaw() const
     {
        return reinterpret_cast<void **>( _cur );
     }

  private:
     int **_cur;
  };

  typedef EventIterator event_iterator;

public:
   static ProfData *getProfilePtr( void *args, size_t evOffset )
   {
      uint8_t *metaBase = static_cast<uint8_t *>( args ) + evOffset;
      EvData *meta = reinterpret_cast<EvData *>( metaBase );

      uint8_t *evBase = reinterpret_cast<uint8_t *>( meta ) +
                        sizeof( EvData );

      int **profBase = reinterpret_cast<int **>( evBase ) + meta->_evCount;

      return reinterpret_cast<ProfData *>( profBase );
   }

protected:
   OCLDD ( bool profiled = false ) : DD( &OCLDev ),
                                     _profiled( profiled ) { }

   // The state of DD class is the Device associated to this DD, and it is
   // represented using a pointer. Since the cluster backend uses copy
   // constructors to rebuild DD on destination hosts, we have to force the
   // pointer in Device to be correctly set -- force here to always point to
   // the local instance of OCLDev.
   OCLDD( const OCLDD &dd ) : DD( &OCLDev ),
                              _profiled( dd._profiled ) { }

public:
   virtual void lazyInit ( WorkDescriptor &wd,
                           bool isUserLevelThread,
                           WorkDescriptor *previous = NULL )
   {
      if( isUserLevelThread )
        fatal0( "Cannot handle user level threads" );
   }

   bool isProfiled() const { return _profiled; }

protected:
   void dump( void *data, size_t evOffset );

private:
   bool _profiled;
};

// Special factory used to provide opencl support to mercurium
//
// +------------------+ -+-
// | Sources          |  |
// | Kernel name      |  |
// | Number of args   |  |
// +------------------+  | Custom section
// | Arg 1            |  |      <--Size decoration is now 0|1 on last bit
// | ...              |  |         1 means buffer arg, 0 inmediate     
// | Arg m            |  |         
// +------------------+ -+-
// | Number of events |
// +------------------+
// | Input event 1    |
// | ...              |
// | Input event n    |
// +------------------+
// | Output event     |
// +------------------+ -+-
// | Start tick       |  | Profiling section
// | End tick         |  | (optional)
// +------------------+ -+-
//
// An iterator is available to iterate over arguments.


class OCLNDRangeKernelStarSSDD : public OCLDD {
public:
   static const size_t MaxWorkDim = 4;

public:
   struct Data
   {
      const char *_programSrcs;
      const char *_kernName;
      const char *_compilerOptions;

      unsigned _argsCount;
   } __attribute__((packed));

   struct Arg
   {
      size_t _size;
      void *_ptr;
   } __attribute__((packed));

   class ArgsIterator
   {
   public:
      static ArgsIterator begin( void *args )
      {
         Data *meta = reinterpret_cast<Data *>( args );

         uint8_t *argsBase = reinterpret_cast<uint8_t *>( meta ) +
                             sizeof( Data );

         return ArgsIterator( reinterpret_cast<Arg *>( argsBase ) );
      }

      static ArgsIterator end( void *args )
      {
         Data *meta = reinterpret_cast<Data *>( args );

         uint8_t *argsBase = reinterpret_cast<uint8_t *>( meta ) +
                             sizeof( Data );

         return ArgsIterator( reinterpret_cast<Arg *>( argsBase ) +
                              meta->_argsCount );
      }

   public:
      ArgsIterator() : _cur( NULL ) { }

   private:
      ArgsIterator( Arg *args ) : _cur( args ) { }

   public:
      bool operator==( const ArgsIterator &iter ) const
      {
         return _cur == iter._cur;
      }

      bool operator!=( const ArgsIterator &iter ) const
      {
         return _cur != iter._cur;
      }

      Arg &operator*() const { return *_cur; }

      Arg *operator->() const { return _cur; }

      ArgsIterator &operator++()
      {
         ++_cur; return *this;
      }

      ArgsIterator operator++( int ign )
      {
         ArgsIterator iter = *this; ++(*this); return iter;
      }

      ArgsIterator operator-( int offset ) const
      {
         return ArgsIterator( _cur - offset );
      }

      ArgsIterator operator+( int offset ) const
      {
        return ArgsIterator( _cur + offset );
      }

      ptrdiff_t operator-( ArgsIterator iter ) const
      {
        return _cur - iter._cur;
      }

  public:
     void **getRaw() const
     {
        return reinterpret_cast<void **>( _cur );
     }

   private:
      Arg *_cur;
   };

   typedef ArgsIterator arg_iterator;

public:
    static size_t eventsOffset( void *args )
    {
       Data *meta = reinterpret_cast<Data *>( args );

       return sizeof( Data ) + meta->_argsCount * sizeof( Arg );
    }

public:
   OCLNDRangeKernelStarSSDD( size_t workDim,
                       size_t *globalWorkOffset,
                       size_t *globalWorkSize,
                       size_t *localWorkSize,
                       void * oclData,
                       bool profiled = false );

public:
   virtual DeviceData *copyTo ( void *addr )
   {
      return new ( addr ) OCLNDRangeKernelStarSSDD( *this );
   }

   virtual size_t size()
   {
      return sizeof( OCLNDRangeKernelStarSSDD );
   }

public:
   size_t getWorkDim() { return _workDim; }
   size_t *getGlobalWorkOffset() { return _globalWorkOffset; }
   size_t *getGlobalWorkSize() { return _globalWorkSize; }
   size_t *getLocalWorkSize() { return _localWorkSize; }
   void *getOpenCLData() { return _oclData; }
   void deleteData(void *data);

public:
   void dump( void *data );

private:
   size_t _workDim;
   size_t _globalWorkOffset[MaxWorkDim];
   size_t _globalWorkSize[MaxWorkDim];
   size_t _localWorkSize[MaxWorkDim];
   void * _oclData;

   friend std::ostream &operator<<( std::ostream &, OCLNDRangeKernelStarSSDD & );
};

} // End namespace ext.
} // End namespace nanos.

#endif // _NANOS_OCL_WD
