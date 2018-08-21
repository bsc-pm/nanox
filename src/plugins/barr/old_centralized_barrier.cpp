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

#include "barrier.hpp"
#include "system.hpp"
#include "atomic.hpp"
#include "schedule.hpp"
#include "plugin.hpp"
#include "synchronizedcondition.hpp"

namespace nanos {
   namespace ext {

      /*! \class OldCentralizedBarrier
       *  \brief implements a single semaphore barrier
       */
      class OldCentralizedBarrier: public Barrier
      {

         private:
            Atomic<int> _sem;
            Atomic<bool> _flag;
            MultipleSyncCond<EqualConditionChecker<bool> > _syncCondTrue;
            MultipleSyncCond<EqualConditionChecker<bool> > _syncCondFalse;
            int _numParticipants;

         public:
            OldCentralizedBarrier () : Barrier(), _sem(0), _flag(false),
               _syncCondTrue( EqualConditionChecker<bool>( &(_flag.override()), true ), 1 ),
               _syncCondFalse( EqualConditionChecker<bool>( &(_flag.override()), false ), 1 ) {}
            OldCentralizedBarrier ( const OldCentralizedBarrier& orig ) : Barrier(orig), _sem(0), _flag(false),
               _syncCondTrue( EqualConditionChecker<bool>( &(_flag.override()), true ), orig._numParticipants ),
               _syncCondFalse( EqualConditionChecker<bool>( &(_flag.override()), false ), orig._numParticipants )
               { init( orig._numParticipants ); }

            const OldCentralizedBarrier & operator= ( const OldCentralizedBarrier & barrier );

            virtual ~OldCentralizedBarrier() { }

            void init ( int numParticipants );
            void resize ( int numThreads );

            void barrier ( int participant );
      };

      const OldCentralizedBarrier & OldCentralizedBarrier::operator= ( const OldCentralizedBarrier & orig )
      {
         // self-assignment
         if ( &orig == this ) return *this;

         Barrier::operator=(orig);
         _sem = 0;
         _flag = false;

         if ( orig._numParticipants != _numParticipants )
            resize(orig._numParticipants);
         
         return *this;
      }

      void OldCentralizedBarrier::init( int numParticipants ) 
      {
         _numParticipants = numParticipants;
         _syncCondTrue.resize( numParticipants );
         _syncCondFalse.resize( numParticipants );
      }

      void OldCentralizedBarrier::resize( int numParticipants ) 
      {
         _numParticipants = numParticipants;
         _syncCondTrue.resize( numParticipants );
         _syncCondFalse.resize( numParticipants );
      }


      void OldCentralizedBarrier::barrier( int participant )
      {
         int val;


         val = ++_sem;

         /* the last process incrementing the semaphore sets the flag
         releasing all other threads waiting in the next block */
         if ( val == _numParticipants ) {
            _flag=true;
            // FIXME: reduction here and remove 2nd phase?
            _syncCondTrue.signal();
            computeVectorReductions();

         } else {
            _syncCondTrue.wait();
         }

         val = --_sem;

         /* the last thread decrementing the sem for the second time resets the flag.
         A thread passing in the next barrier will be blocked until this is performed */
         if ( val == 0 ) {
            _flag=false;
            _syncCondFalse.signal();
         } else {
            _syncCondFalse.wait();
         }
      }


      static Barrier * createOldCentralizedBarrier()
      {
         return NEW OldCentralizedBarrier();
      }


      /*! \class OldCentralizedBarrierPlugin
       *  \brief plugin of the related OldCentralizedBarrier class
       *  \see OldCentralizedBarrier
       */
      class OldCentralizedBarrierPlugin : public Plugin
      {

         public:
            OldCentralizedBarrierPlugin() : Plugin( "OldCentralized Barrier Plugin",1 ) {}

            virtual void config( Config &cfg ) {}

            virtual void init() {
               sys.setDefaultBarrFactory( createOldCentralizedBarrier );
            }
      };

   }
}

DECLARE_PLUGIN("barr-old-centralized",nanos::ext::OldCentralizedBarrierPlugin);
