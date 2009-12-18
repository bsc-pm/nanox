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

#include "barrier.hpp"
#include "system.hpp"
#include "atomic.hpp"
#include "schedule.hpp"
#include "plugin.hpp"
#include "synchronizedcondition.hpp"

namespace nanos {
   namespace ext {

      /*! \class CentralizedBarrier
       *  \brief implements a single semaphore barrier
       */
      class CentralizedBarrier: public Barrier
      {

         private:
            Atomic<int> _sem;
            Atomic<MultipleSyncCond<int> *> _syncCond1;
            Atomic<MultipleSyncCond<int> *> _syncCond2;
            int _numParticipants;

         public:
            CentralizedBarrier () : Barrier(), _sem(0), _syncCond1(NULL), _syncCond2(NULL) {}
            CentralizedBarrier ( const CentralizedBarrier& barrier ) : Barrier(barrier),_sem(0),_syncCond1(NULL),_syncCond2(NULL)
               { init( barrier._numParticipants ); }

            const CentralizedBarrier & operator= ( const CentralizedBarrier & barrier );

            virtual ~CentralizedBarrier() { }

            void init ( int numParticipants );
            void resize ( int numThreads );

            void barrier ( int participant );
      };

      const CentralizedBarrier & CentralizedBarrier::operator= ( const CentralizedBarrier & barrier )
      {
         // self-assignment
         if ( &barrier == this ) return *this;

         Barrier::operator=(barrier);
         _sem = 0;
         _syncCond1 = NULL;
         _syncCond2 = NULL;

         if ( barrier._numParticipants != _numParticipants )
            resize(barrier._numParticipants);
         
         return *this;
      }

      void CentralizedBarrier::init( int numParticipants ) 
      {
         _numParticipants = numParticipants;
      }

      void CentralizedBarrier::resize( int numParticipants ) 
      {
         _numParticipants = numParticipants;
      }


      void CentralizedBarrier::barrier( int participant )
      {
        /*
         */
         if ( _syncCond1 == NULL ) {
            MultipleSyncCond<int> *tmp = new MultipleSyncCond<int>(&_sem.override(), _numParticipants);
            if ( !_syncCond1.cswap( NULL, tmp ) ) {
               delete (tmp);
            }
         }

         ++_sem;
         _syncCond1.override()->wait();

        /*
         */
         if ( _syncCond2 == NULL ) {
            MultipleSyncCond<int> *tmp = new MultipleSyncCond<int>(&_sem.override(), 0);
            if ( !_syncCond2.cswap( NULL, tmp ) ) {
               delete (tmp);
            }
         }
         --_sem;
         _syncCond2.override()->wait();
      }


      static Barrier * createCentralizedBarrier()
      {
         return new CentralizedBarrier();
      }


      /*! \class CentralizedBarrierPlugin
       *  \brief plugin of the related centralizedBarrier class
       *  \see centralizedBarrier
       */
      class CentralizedBarrierPlugin : public Plugin
      {

         public:
            CentralizedBarrierPlugin() : Plugin( "Centralized Barrier Plugin",1 ) {}

            virtual void init() {
               sys.setDefaultBarrFactory( createCentralizedBarrier );
            }
      };

   }
}

nanos::ext::CentralizedBarrierPlugin NanosXPlugin;

