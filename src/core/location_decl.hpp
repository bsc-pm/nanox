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

} // namespace nanos

#endif /* LOCATION_DECL_H */
