/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

#include "nanos-int.h"

#ifndef _NANOS_WORK_SHARING_H
#define _NANOS_WORK_SHARING_H

namespace nanos {

   class WorkSharing {
      private:
      public:
         WorkSharing () {}
         virtual ~WorkSharing () {}

         /*! \brief create a loop descriptor
          *  
          *  \return only one thread per loop will get 'true' (single like behaviour)
          */
         virtual bool create( nanos_ws_desc_t **wsd, nanos_ws_info_t *info ) = 0;

         /*! \brief Get next chunk of iterations
          *
          *  \return if there are more iterations to execute
          */
         virtual void nextItem( nanos_ws_desc_t *wsd, nanos_ws_item_t *wsi ) = 0 ;

         /*! \brief Duplicates a WorkSharing Descriptor
          */
         virtual void duplicateWS ( nanos_ws_desc_t *orig, nanos_ws_desc_t **copy) = 0;


   };

} // namespace nanos

#endif

// XXX: EOF - to remove after completing design
// XXX: compiler transformation proposal
#if 0

   xxxxx -> scheduler policy

   // outline transformation

   void ol_for_code ( void * args ) {

      nanos_loop_info_t ld = (nanos_loop_desc_t *) args; 

      int lower = loc->

      while ( nanos_loop_next_iters ( ld, &lower, &upper, &step, &last ) ) {

      }

      nanos_team_barrier(); // optional (if no 'nowait' clause) --> nanos_omp_barrier() el comportamiento hara que los implicitos tambien esperen tareas
   }

   // inline transformation

   static nanos_worksharing_t xxxxx_worksharing_for
   if ( !xxxxx_worksharing_for ) xxxxx_worksharing_for = nanos_find_worksharing("xxxxx_for");

   nanos_loop_desc_t *ld;
   single = nanos_create_loop ( xxxxx_worksharing_for, ld ); // no one will continue until ld has been initilized

   if ( (single) && (nanos_get_nonimplicit_threads() > 0) ) {
      static nanos_slicer_t xxxxx_slicer_ws_for = 0; 
      if ( !xxxxx_slicer_ws_for ) xxxxx_slicer_ws_for = nanos_find_slicer("xxxxx_for");
      
      err = nanos_create_sliced_wd ( ... );
      // Initialize data
      err = nanos_submit(wd, ... );
   }

   ol_for_code ( ... ):
   // while ( nanos_loop_next_iters ( ld, &lower, &upper, &step, &last ) ) {

   }

   nanos_team_barrier(); // optional (if no 'nowait' clause) --> nanos_omp_barrier() el comportamiento hara que los implicitos tambien esperen tareas


NOTAS:
  - una estructura worksharing se distribuye entre todos los miembros de un team
  - entre los dos modelos de ejecucion actuales: omp y ompss podriamos distinguir las siguientes estructuras de teams:
     - omp : n threads implicitos y 0 threads no implicitios
     - ompss: 1 thread implicito y n-1 threads no implicitios

  - la idea basica de la transformacion consiste en distinguir 2 tipos de threads que participan en la ejecucion de una
    estructura worksharing. Primero los threads implicitios (aquellos que no solo forman parte del team, sino que ademas
    ejecutan el mismo codigo). Segundo el resto de threads del team (los no implicitos).
  - El codigo creara una transformacion  inline capaz de distribuir la region worksharing entre los threads implicitos,
    y tareas sliceables para el resto
    de threads del team.
  - loop descriptor puede ser (en funcion del scheduler) una estructura privada (static) o compartida (dynamic o guided).
    en todos los casos debe contener la informacion global del loop, ya que en determinados casos podrian crearse
    en la segunda fase de la transformacion del 'for' slicers que deberia poder reconstruir la informacion del loop

  - slicer worksharing no es exactamente el repeat N, ya que deberia crear tantos wd como threads no implicitos tenga
    el team y realizar el tie_to a ese trhread para despues ejecutar el codigo del ws. La idea consiste en exentender
    el team para ejecutar un codigo

  - extra_team_barrier deberia extenderse tambien a los threads corriendo slicers ( barrier de implicit y no implicit)

  - servicios nuevos de la API:
    - nanos_find_worksharing()
    - nanos_create_loop() // funcion especifica worksharing
    - nanos_loop_next_chunk() // funcion especifica worksharing
    - nanos_get_nonimplicit_threads();


#endif
