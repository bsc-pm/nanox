#ifndef LOCATION_DECL_H
#define LOCATION_DECL_H

//#include <vector>

namespace nanos {
   class Location {
         unsigned int _clusterNode;
         unsigned int _numaNode;
         unsigned int _socket;
         bool         _inNumaNode;
         bool         _inSocket;

         Location ();
         Location ( Location const &l );
         Location &operator=( Location const &l );
      public:
         Location ( unsigned int clusterNode, unsigned int numaNode, bool inNumaNode, unsigned int socket, bool inSocket );
         unsigned int getClusterNode() const;
         //void setClusterNode( unsigned int clusterNode );
         unsigned int getNumaNode() const;
         //void setNumaNode( unsigned int numaNode );
         bool isInNumaNode() const;
         unsigned int getSocket() const;
         //void setSocket( unsigned int socket );
         bool isInSocket() const;
   };

   /*
   class LocationDirectory {
      std::vector< Location > _locations;

      LocationDirectory( LocationDirectory const &ld );
      LocationDirectory &operator=( LocationDirectory const &ld );

      public:
         LocationDirectory();
         void initialize( unsigned int numLocs );
         Location &operator[]( unsigned int locationId );
   };
   */
}

#endif /* LOCATION_DECL_H */
