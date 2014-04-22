#ifndef LOCATION_DECL_H
#define LOCATION_DECL_H

#include <vector>

namespace nanos {
   class Location {
      unsigned int _nodeId;
      unsigned int _memorySpaceId;
      unsigned int _socketId;
      unsigned int _coreId;

      public:
         Location ();
         Location ( Location const &l );
         Location &operator=( Location const &l );
         unsigned int getNodeId() const;
         void setNodeId( unsigned int nodeId );
         unsigned int getMemorySpaceId() const;
         void setMemorySpaceId( unsigned int memId );
         unsigned int getSocketId() const;
         void setSocketId( unsigned int socketId );
         unsigned int getCoreId() const;
         void setCoreId( unsigned int coreId );
   };

   class LocationDirectory {
      std::vector< Location > _locations;

      LocationDirectory( LocationDirectory const &ld );
      LocationDirectory &operator=( LocationDirectory const &ld );

      public:
         LocationDirectory();
         void initialize( unsigned int numLocs );
         Location &operator[]( unsigned int locationId );
   };
}

#endif /* LOCATION_DECL_H */
