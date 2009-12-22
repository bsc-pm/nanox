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

#include <assert.h>
#include <vector>
#include <math.h>

#include "barrier.hpp"
#include "system.hpp"
#include "atomic.hpp"
#include "plugin.hpp"
#include "synchronizedcondition.hpp"

using namespace std;

namespace nanos {
   namespace ext {

      /*!
       *  \class disseminationBarrier
       *  \brief implements a barrier according to the dissemination algorithm
       */

      class DisseminationBarrier: public Barrier
      {

         private:
            /*! the semaphores are implemented with a vector of atomic integers, because their number can change in successive
               invocations of the barrier method */
            vector<Atomic<int> > _semaphores;
            vector<SingleSyncCond<int> *> _syncConds1;
            vector<SingleSyncCond<int> *> _syncConds2;
            Atomic<unsigned int> _ticket;

            /*! number of participants: can change dynamically */
            int _numParticipants;

            /*! number of steps of the algorithm, equal to the logarithm of the number of participants */
            int _q;

         public:
            /*! \warning the creation of the pthread_barrier_t variable will be performed when the barrier function is invoked
                   because only at that time we exectly know the number of participants (which is dynamic, as in a team
                   threads can dynamically enter and exit)
             */
            DisseminationBarrier() { }

            void init ( int numParticipants );
            void resize ( int numParticipants );

            void barrier( int numParticipants );

            ~DisseminationBarrier() { }
      };

      void DisseminationBarrier::init( int numParticipants ) 
      {
         _numParticipants = numParticipants;

         _semaphores.resize( _numParticipants );
         _syncConds1.resize( _numParticipants );
         _syncConds2.resize( _numParticipants );

         _ticket = 0;

         for ( int i = 0; i < _numParticipants; i++ ) {
            _semaphores[i] = 0;
            _syncConds1[i] = new SingleSyncCond<int>( &_semaphores[i].override(), 0);
            _syncConds2[i] = new SingleSyncCond<int>( &_semaphores[i].override(), -1);
         }

         /*! we can compute the number of steps of the algorithm */
         _q = ( int ) ceil( log2( ( double ) _numParticipants ) );

         std::cout << "Num part = " << _numParticipants << " and q = " << _q << std::endl;
      }

      void DisseminationBarrier::resize( int numParticipants ) 
      {
         _numParticipants = numParticipants;

         _semaphores.resize( _numParticipants );
         _syncConds1.resize( _numParticipants );
         _syncConds2.resize( _numParticipants );

         _ticket = 0;

         for ( int i = 0; i < _numParticipants; i++ ) {
            _semaphores[i] = 0;
            _syncConds1[i] = new SingleSyncCond<int>( &_semaphores[i].override(), 0);
            _syncConds2[i] = new SingleSyncCond<int>( &_semaphores[i].override(), -1);
         }

         /*! we can re-compute the number of steps of the algorithm */
         _q = ( int ) ceil( log2( ( double ) _numParticipants ) );
      }


      void DisseminationBarrier::barrier( int numParticipants )
      {
//         int myID = myThread->getId();
warning("got ticket " << _ticket.override());
         int myID = (_ticket++)%_numParticipants;

         for ( int s = 0; s < _q; s++ ) {
            //compute the current step neighbour id
            int toSign = ( myID + ( int ) pow( 2,s ) ) % _numParticipants;

warning(" waiting for "<< toSign << " value is " << _semaphores[toSign].override() << " expecting 0");
            _syncConds1[toSign]->wait();
            ( _semaphores[toSign] )--;
warning(" signaling "<< toSign << " value set to " << _semaphores[toSign].override());
            _syncConds2[toSign]->signal();

warning(" 2 waiting for "<< myID << " value is " << _semaphores[myID].override() << " expecting -1");
            _syncConds2[myID]->wait();
            ( _semaphores[myID] )++;
warning(" signaling "<< myID << " value set to " << _semaphores[myID].override());
            _syncConds1[myID]->signal();
/*
            //wait for the neighbour sem to reach the previous value
warning(" waiting for "<< toSign << " sem value: "<< _semaphores[toSign].override());
            Scheduler::blockOnCondition( &_semaphores[toSign].override(), 0 );
            ( _semaphores[toSign] )--;
            / *!
             *  Wait for the semaphore to be signaled for this round
             *  (check if it reached the number of step signals (+1 because the for starts from 0 
             *  for a correct computation of neighbours)
             * /
warning(" myself, for "<< myID << " sem value: "<< _semaphores[myID].override());
            Scheduler::blockOnCondition( &_semaphores[myID].override(), -1 );

            ( _semaphores[myID] )++;
*/
         }

         /*! at the end of the protocol, we are guaranteed that the semaphores are all 0 */
      }

      static Barrier * createDisseminationBarrier()
      {
         return new DisseminationBarrier();
      }

      /*! \class DisseminationBarrierPlugin
       *  \brief plugin of the related disseminationBarrier class
       *  \see disseminationBarrier
       */

      class DisseminationBarrierPlugin : public Plugin
      {

         public:
            DisseminationBarrierPlugin() : Plugin( "Dissemination Barrier Plugin",1 ) {}

            virtual void init() {
               sys.setDefaultBarrFactory( createDisseminationBarrier );
            }
      };

   }
}

nanos::ext::DisseminationBarrierPlugin NanosXPlugin;
