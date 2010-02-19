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

#include "schedule.hpp"
#include "wddeque.hpp"
#include "plugin.hpp"
#include "system.hpp"
#include "config.hpp"

namespace nanos {
   namespace ext {

      class WFData : public SchedulingData
      {

         friend class WFPolicy; //in this way, the policy can access the readyQueue

         private:
            WDDeque _readyQueue;

            WFData ( const WFData & );
            const WFData operator= ( const WFData & );

         public:
            // constructor
            WFData( int id=0 ) : SchedulingData( id ) {}

            // destructor
            ~WFData()
            {
                ensure(_readyQueue.empty(),"Destroying non-empty wdqueue");
            }
      };

      class WFPolicy : public SchedulingGroup
      {
         private:
            WFPolicy ( const WFPolicy & );
            const WFPolicy operator= ( const WFPolicy & );

         public:
            typedef enum { FIFO, LIFO } QueuePolicy;

            // Should these be private?
            static bool          _noStealParent;
            static QueuePolicy   _localPolicy;
            static QueuePolicy   _stealPolicy;

            // constructor
            WFPolicy() : SchedulingGroup( "wf-steal-sch" ) {} 
            WFPolicy( int groupsize ) : SchedulingGroup( "wf-steal-sch", groupsize ) {}

            virtual ~WFPolicy() {}

            virtual WD *atCreation ( BaseThread *thread, WD &newWD );
            virtual WD *atIdle ( BaseThread *thread );
            virtual void queue ( BaseThread *thread, WD &wd );
            virtual SchedulingData * createMemberData ( BaseThread &thread );

            /*! \brief Extracts a WD from the queue either from the beginning or the end of the queue
             *
             *  This function allows to simplify the code to extract code from the queues. It's a wrapper around the WDDeque
             *  functions with the actual function chosen with the policy argument.
             *
             *   \param [inout] q The queue from we want to extract a WD
             *   \param [in] policy Either FIFO/LIFO to specify if we extract from the beginning or the end of the queue
             *   \param [in] thread The thread trying to extract the thread
             *   \returns either a WD if one was available in the queues or NULL
             *   \sa WDDeque::pop_front, WDDeque::pop_back
             */
            WD * pop ( WDDeque &q, QueuePolicy policy, BaseThread *thread );
      };

      bool WFPolicy::_noStealParent = false;
      WFPolicy::QueuePolicy WFPolicy::_localPolicy = WFPolicy::FIFO;
      WFPolicy::QueuePolicy WFPolicy::_stealPolicy = WFPolicy::FIFO;

      void WFPolicy::queue ( BaseThread *thread, WD &wd )
      {
         WFData *data = ( WFData * ) thread->getSchedulingData();
         data->_readyQueue.push_front( &wd );
      }

      WD * WFPolicy::atCreation ( BaseThread *thread, WD &newWD )
      {
         //NEW: now it does not enqueue the created task, but it moves down to the generated son: DEPTH-FIRST
         return &newWD;
      }

      WD * WFPolicy::pop ( WDDeque &q, QueuePolicy policy, BaseThread *thread )
      {
         return policy == LIFO  ? q.pop_front(thread) : q.pop_back(thread);
      }

      WD * WFPolicy::atIdle ( BaseThread *thread )
      {
         WorkDescriptor * wd = NULL;

         WFData *data = ( WFData * ) thread->getSchedulingData();

         if ( ( wd = pop(data->_readyQueue, _localPolicy, thread) ) != NULL ) {
            return wd;
         } else {
            if ( _noStealParent == false ) {
               if ( ( wd = ( thread->getCurrentWD() )->getParent() ) != NULL ) {
                  /*!
                   * removing it from the queue. Try to remove from one queue: if someone move it,
                   * stop looking to avoid ping-pongs.
                   */
                  if ( wd->isEnqueued()  && ( !wd->isTied() || wd->isTiedTo() == thread ) ) {
                     //not in queue = in execution, in queue = not in execution
                     if ( wd->getMyQueue()->removeWD( wd ) == true ) { //found it!
                        return wd;
                     }
                  }
               }
            }

            /*! next: steal from other queues */
            int newposition = data->getSchId();
            wd = NULL;

            while ( wd == NULL ) {
               newposition = ( newposition + 1 ) % getSize();
               if ( newposition != data->getSchId() ) {
                  wd = pop( (( WFData * ) getMemberData ( newposition ))->_readyQueue, _stealPolicy, thread );
               }

            }
            return wd;
         }
      }

      SchedulingData * WFPolicy::createMemberData ( BaseThread &thread )
      {
         return new WFData();
      }

      // Factories

      static SchedulingGroup * createWFPolicy ( int groupsize )
      {
         return new WFPolicy( groupsize );
      }


      class WFSchedPlugin : public Plugin
      {

         public:
            WFSchedPlugin() : Plugin( "WF scheduling Plugin",1 ) {}

            virtual void config( Config& config )
            {
               config.setOptionsSection( "Wf module", new std::string("Width-first scheduling module") );
               //BUG: If defining local policy or steal policy the command line option *must not* include the = between
               //the option name and the value, but a space
               config.registerConfigOption ( "wf-no-steal-parent", new Config::FlagOption( WFPolicy::_noStealParent ), "Do not steal parent" );
               config.registerArgOption ( "wf-no-steal-parent", "wf-no-steal-parent" );

               Config::MapVar<WFPolicy::QueuePolicy> queuePolicyLocalConfig ( WFPolicy::_localPolicy );
               queuePolicyLocalConfig.addOption ( "FIFO", WFPolicy::FIFO ).addOption ( "LIFO", WFPolicy::LIFO );
               config.registerConfigOption ( "wf-local-policy", &queuePolicyLocalConfig, "WF local queue policy");
               config.registerArgOption ( "wf-local-policy", "wf-local-policy" );

               Config::MapVar<WFPolicy::QueuePolicy> queuePolicyStealConfig ( WFPolicy::_localPolicy );
               queuePolicyStealConfig.addOption ( "FIFO", WFPolicy::FIFO ).addOption ( "LIFO", WFPolicy::LIFO );
               config.registerConfigOption ( "wf-steal-policy", &queuePolicyStealConfig, "WF steal queue policy");
               config.registerArgOption ( "wf-steal-policy", "wf-steal-policy" );

               config.init();
            }

            virtual void init() {
               sys.setDefaultSGFactory( createWFPolicy );
            }
      };

   }
}

nanos::ext::WFSchedPlugin NanosXPlugin;


