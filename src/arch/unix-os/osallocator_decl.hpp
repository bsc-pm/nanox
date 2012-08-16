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
      int tryAlloc( uintptr_t addr, size_t len ) const; 
      void readeMaps();

      void print_current_maps(void) const;
      void print_parsed_maps() const; 
      void print_parsed_maps_full() const; 

   public:
      void *allocate( size_t len ); 
};

}
