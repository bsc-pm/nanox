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

namespace nanos {

class OSAllocator {

   private:
      struct OSMemoryMap
      {
         uintptr_t start;
         uintptr_t end;
         int prots;
         OSMemoryMap( ) : start( 0 ), end( 0 ), prots( 0 ) {}
         OSMemoryMap( uintptr_t st, uintptr_t en, int pr ) : start( st ), end( en ), prots( pr ) {}
         OSMemoryMap( OSMemoryMap const &map ) : start( map.start ), end( map.end ), prots( map.prots ) {}
      };

      std::list< OSMemoryMap > processMaps;
      std::list< OSMemoryMap > freeMaps;

      size_t computeFreeSpace( uintptr_t start, uintptr_t end, char &unit ) const;
      uintptr_t lookForAlignedAddress( size_t len ) const; 
      int tryAlloc( uintptr_t addr, size_t len, int flags ) const; 
      void readeMaps();

      void print_current_maps(void) const;
      void print_parsed_maps() const; 
      void print_parsed_maps_full() const; 
      void *_allocate( size_t len, bool none ); 

   public:
      void *allocate( size_t len ); 
      void *allocate_none( size_t len ); 
};

} // namespace nanos
