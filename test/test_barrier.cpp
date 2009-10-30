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

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include <errno.h>
#include<vector>
#include<math.h>

#include "barrier.hpp"
#include "atomic.hpp"
#include "schedule.hpp"


using namespace nanos;

using namespace std;

void debug_c( const char * str, int x )
{
   pthread_t tid = pthread_self();
   printf( "Thread %lu: %s %x\n", tid, str, x );
}



//Put Here the Barrier Implementation for pre-test


class disseminationBarrier: public Barrier
{

   private:
      vector<Atomic<int> > semaphores;
      int q; //q is the log of threadNum

   public:
      disseminationBarrier( int numP );
      void init();
      void barrier();
      void barrier( int id );
};

//TODO: number of threads must be a power of 2: remove this hypothesis
disseminationBarrier::disseminationBarrier( int numP ): Barrier( numP )
{
   assert( numP > 0 );

   std::cout << "computing q as log(" << numP << ")" << std::endl;
   q = ( int ) ceil( log2( numP ) );

   /*! semaphores are automatically initialized to their default value (\see Atomic)*/
   semaphores.resize( numP );
}

void disseminationBarrier::init() {}

void disseminationBarrier::barrier()
{
   //for now, suppose that we can obtain an ID
   int myID = -1;

   for ( int s = 0; s < q; s++ ) {
      int toSign = ( myID + ( int ) pow( 2,s ) ) % numParticipants;
      ( semaphores[toSign] )--;
      //std::cout << "Thread " <<  << ": decrementing the sem of " << toSign << std::endl;

      while ( semaphores[myID] != 0 ) {}

      if ( s < q ) {
         //reset the semaphore for the next round
         semaphores[myID]++; //notice that we allow another thread to signal it before we reset it (not set to 1!)
      }
   }

   //reset the semaphore for the next barrier
   //warning: as we do not have an atomic assignement, we use the substraction operation
   semaphores[myID]-1;
}


void disseminationBarrier::barrier( int myID )
{
   std::cout << "q = " << q << std::endl;

   for ( int s = 0; s < q; s++ ) {
      //int signal_val = s+1;
      std::cout << "Thread " <<  myID << ": step " << s << std::endl;
      fflush( stdout );

      int toSign = ( myID + ( int ) pow( 2,s ) ) % numParticipants;
      //first wait for the neighbour sem to reach the previous value
      Scheduler::blockOnCondition( &semaphores[toSign].override(), 0 );
      ( semaphores[toSign] )--;

      std::cout << "Thread " <<  myID << ": incremented the sem of " << toSign << std::endl;
      fflush( stdout );
      /*!
         Wait for the semaphore to be signaled for this round
         (check if it reached the number of step signals (+1 because the for starts from 0 for a correct computation of neighbours)
       */
      Scheduler::blockOnCondition( &semaphores[myID].override(), -1 );
      ( semaphores[myID] ) + 1;

      std::cout << "Thread " <<  myID << ": now my sem is "<< semaphores[myID] << std::endl;
      fflush( stdout );
   }

   /*! Reset the semamphore for the next barrier: \warning as we are not supported by atomic resetting, we need to use the substraction of the reached value */
   //(semaphores[myID]) - q;
}




Barrier * createDisseminationBarrier( int numP )
{
   return new disseminationBarrier( numP );
}

///////*************





disseminationBarrier * bar;

void * barrieringThread( void * arg ) ;

int main ( int argc, char ** argv )
{
   int numT = 10, i;
   pthread_t * tids;

   if ( argc > 2 ) printf( "Usage: test_barrier <numThreads>\n" );

   if ( argc == 2 ) numT = strtol( argv[1], NULL, 10 );

   tids = ( pthread_t * ) calloc( numT, sizeof( pthread_t ) );

   bar = new disseminationBarrier( numT );

   std::cout << "Main: running " << numT << " threads"  << std::endl;

   for ( i = 0; i < numT; i++ ) {
      int *id = ( int * ) calloc( 1, sizeof( int ) );
      *id = i;
      int err = pthread_create( &tids[i], NULL, barrieringThread, id );

      if ( err != 0 ) {
         switch ( err ) {

            case EAGAIN:
               std::cout << "EAGAIN" << std::endl;
               break;

            case EINVAL:
               std::cout << "EINVAL" << std::endl;
               break;

            case EPERM:
               std::cout << "EPERM" << std::endl;
         }
      }
   }

   for ( i = 0; i < numT; i++ ) {
      pthread_join( tids[i], NULL );
   }

   std::cout << "Main: after the join"  << std::endl;


}



//thread function, performing a barrier. Users can specify a number of seconds in the arg argument before the barrier function is actually invoked
void * barrieringThread( void * arg )
{
   int myID = -1;

   if ( arg != NULL )  myID = ( int ) ( *( int * )arg );

   std::cout << "I am thread " << myID << std::endl;

//    fflush(stdout);
//
//    std::cout << "Thread " << myID << "calling barrier " << std::endl;
//    fflush(stdout);

   bar->barrier( myID );

//    std::cout << "Thread " << myID << "after barrier " << std::endl;
//    fflush(stdout);

//    std::cout << "calling barrier 2"<< std::endl;
//
//    bar->barrier();
//
//    std::cout << "after barrier 2" << std::endl;

   pthread_exit( 0 );
}

