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

#ifndef _NANOS_DEPENDENCIES_DOMAIN
#define _NANOS_DEPENDENCIES_DOMAIN
#include <stdlib.h>
#include <map>
#include <list>
#include <vector>
#include "dependableobject.hpp"
#include "trackableobject.hpp"
#include "dependency.hpp"


namespace nanos
{

  /*! \class DependenciesDomain
   *  \brief Each domain is an independent context in which dependencies between DependableObject are managed
   */
   class DependenciesDomain
   {
      private:
         /**< Id to be given to the next submitted DependableObject */
         unsigned int _lastDepObjId;
         
         /**< Maps addresses to Trackable objects */
         typedef std::map<void *, TrackableObject*> DepsMap;
         /**< Used to track dependencies between DependableObject */
         DepsMap _addressDependencyMap;

         TrackableObject* lookupDependency ( const Dependency &dep );

         template<typename iterator>
         void submitDependableObjectInternal ( DependableObject &depObj, iterator begin, iterator end );
         
      public:
        /*! \brief Constructor
         */
         DependenciesDomain ( ) :  _lastDepObjId ( 0 ), _addressDependencyMap( ) {}

        /*! \brief Copy Constructor
         */
         DependenciesDomain ( const DependenciesDomain &depDomain ) :  _lastDepObjId ( depDomain._lastDepObjId ), _addressDependencyMap ( depDomain._addressDependencyMap ) {}

        /*! \brief Destructor
         */
         ~DependenciesDomain ( ) { }
         
        /*! \brief Assign operator, can be self-assigned.
         *  \param depDomain another DependenciesDomain
         */
         const DependenciesDomain & operator= ( const DependenciesDomain &depDomain ) {
            if ( this == &depDomain ) return *this;
            _lastDepObjId = depDomain._lastDepObjId;
            _addressDependencyMap = depDomain._addressDependencyMap;
            return *this;
         }
         
        /*! \brief Assigns the DependableObject depObj an id in this domain and adds it to the domains dependency system.
         *  \param depObj DependableObject to be added to the domain.
         *  \param deps List of dependencies to be associated to the Dependable Object.
         *  \sa Dependency DependableObject TrackableObject
         */
         void submitDependableObject ( DependableObject &depObj, const std::vector<Dependency> &deps )
         {
            submitDependableObjectInternal ( depObj, deps.begin(), deps.end() );
         }

        /*! \brief Assigns the DependableObject depObj an id in this domain and adds it to the domains dependency system.
         *  \param depObj DependableObject to be added to the domain.
         *  \param deps List of dependencies to be associated to the Dependable Object.
         *  \param numDeps Number of dependenices in the list.
         *  \sa Dependency DependableObject TrackableObject
         */
         void submitDependableObject ( DependableObject &depObj, Dependency* deps, int numDeps)
         {
            submitDependableObjectInternal ( depObj, deps, &deps[numDeps] );
         }

         void finished ( DependableObject &depObj );


   };

};

#endif

