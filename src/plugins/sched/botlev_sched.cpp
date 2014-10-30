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

#include <iostream>
#include <fstream>
#include <assert.h>

#define SUPPORT_UPDATE 0 

namespace nanos {
   namespace ext {
      static int   updateFreq;
      static int   hpCores;
      static int   numSpins;
      static int   hpFrom;
      static int   hpTo;
      static int   hpSingle;
      static int   steal;
      static int   maxBL;
      static int   strict;
      static unsigned   threshold;
      static int   taskNumber = 0;
//      static unsigned numThreads;
      class BotLevDOData : public DOSchedulerData
      {
         public:
            typedef std::set<BotLevDOData *> predecessors_t;

         private:
            int               _taskNumber;    //! \todo Keep task number just for debugging, remove?
            int               _botLevel;      //! Bottom Level Value
            bool              _isReady;       //! Is the task ready
            Lock              _lock;          //! Structure lock
            WD               *_wd;            //! Related WorkDescriptor
            predecessors_t    _predecessors;  //! List of (dependable object botlev specific data) predecessors 
         public:
            BotLevDOData(int tnum, int blev) : _taskNumber(tnum), _botLevel(blev), _isReady(false), _lock(), _wd(NULL), _predecessors() { }
            ~BotLevDOData() { }
            void reset () { _wd = NULL; }

            int getTaskNumber() const { return _taskNumber; }

            int getBotLevel() const { return _botLevel; }
            bool setBotLevel( int blev )
            {
               if ( blev > _botLevel ) {
                 {
                    LockBlock lock1( _lock );
                    _botLevel = blev;
                 }
                  return true;
               }
               return false;
            }

            bool getReady() const { return _isReady; }
            void setReady() { _isReady = true; }

            WD *getWorkDescriptor() const { return _wd; }
            void setWorkDescriptor(WD *wd) { _wd = wd; }

            std::set<BotLevDOData *> *getPredecessors() { return &_predecessors; }
            void addPredecessor(BotLevDOData *predDod) { _predecessors.insert(predDod); }
      };

      class BotLev : public SchedulePolicy
      {
         public:
            typedef std::stack<BotLevDOData *>   bot_lev_dos_t;
            typedef std::set<DependableObject *> DependableObjectVector; /**< Type vector of successors  */

         private:
            bot_lev_dos_t           _blStack;     //! tasks added, pending having their bottom level updated
            DependableObjectVector  topSuccesors;
            WD * topWD;
            Lock              _lock;          //! Structure lock
            Lock              botLevLock;
            int               maxBotLev;
            int               currMax;
            Lock              fileLock;
            unsigned int      numCritical;

            /** \brief DistributedBF Scheduler data associated to each thread
              *
              */
            struct TeamData : public ScheduleTeamData
            {
               /*! queue of ready tasks to be executed */
               WDPriorityQueue<> *_readyQueues;
               TeamData () : ScheduleTeamData()
               {
                 _readyQueues = NEW WDPriorityQueue<>[3];
                 hpCores = hpTo - hpFrom +1;
//                 fprintf(stderr, "Num Threads in teamData(): %u\n",  myThread->getTeamData()->getTeam()->size());
                 fprintf(stderr, "queue 0 size = %d, queue 1 size = %d\n", (int)_readyQueues[0].size(), (int)_readyQueues[1].size());
                 fprintf(stderr, "queue 0 addr = %p queue 1 addr = %p\n", &_readyQueues[0], &_readyQueues[1]);
                 fprintf(stderr, "numSpins = %d\n", numSpins);
                 fprintf(stderr, "Running with %d HP cores\n", hpCores);
                 fprintf(stderr, "hp from = %d hp to = %d \n", hpFrom, hpTo);
               }
               virtual ~TeamData () { delete[] _readyQueues; }
            };

            /* disable copy and assigment */
            explicit BotLev ( const BotLev & );
            const BotLev & operator= ( const BotLev & );

         typedef enum {
            NANOS_SCHED_BLEV_ZERO,
            NANOS_SCHED_BLEV_ATCREATE,
            NANOS_SCHED_BLEV_ATSUBMIT,
            NANOS_SCHED_BLEV_ATIDLE
         } sched_bf_event_value;

         public:
            // constructor
            BotLev() : SchedulePolicy ( "BotLev" ) {
               topWD = NULL;
               maxBotLev = 0;
               currMax = maxBL;
               threshold = 2;
             //  assert(myThread != NULL);
               //numThreads = myThread->getTeamData()->getTeam()->getNumSupportingThreads();//->size();//sys.getNumNumaNodes();//getNumCreatedPEs();//getMainTeam()->size();//sys.getNumWorkers();
//               fprintf(stderr, "Numworkers = %d\n", (int)numThreads);
               numCritical = 0;
               if(numSpins == 0) numSpins = 300;
            }

            // destructor
            virtual ~BotLev() {}

            virtual size_t getTeamDataSize () const { return sizeof(TeamData); }
            virtual size_t getThreadDataSize () const { return 0; }

            virtual ScheduleTeamData * createTeamData ()
            {
               return NEW TeamData();
            }

            virtual ScheduleThreadData * createThreadData ()
            {
               return 0;
            }

            /*!
            *  \brief Enqueue a work descriptor in the readyQueue of the passed thread
            *  \param thread pointer to the thread to which readyQueue the task must be appended
            *  \param wd a reference to the work descriptor to be enqueued
            *  \sa ThreadData, WD and BaseThread
            */
            virtual void queue ( BaseThread *thread, WD &wd )
            {
#ifdef NANOS_INSTRUMENTATION_ENABLED
                int *criticality = (int*)malloc(sizeof(int)); 
#endif
                TeamData &data = ( TeamData & ) *thread->getTeam()->getScheduleData();
                // Find the priority
                DependableObject *dos = wd.getDOSubmit();
                BotLevDOData *dodata;
                unsigned int priority = 0; //! \todo FIXME priority value is hardcoded
                short qId = -1;
                if ( dos ){ //&& numThreads>1 ) {
                   dodata = (BotLevDOData *)dos->getSchedulerData();
                   dodata->setWorkDescriptor(&wd);
                   priority = dodata->getBotLevel();
                }
                else {
	           if(wd.getDepth() == 0 ) {  //fprintf(stderr, "Queueing implicit or main id %d\n", wd.getId());
                      wd.setPriority(0);
                      data._readyQueues[2].push_back( &wd ); //readyQueue number 2 is for the main and implicit tasks
                   }
                   else {  //in this case numThreads = 1 inserting in queue number 2
                      data._readyQueues[2].push_back( &wd );
                   }
#ifdef NANOS_INSTRUMENTATION_ENABLED
                   *criticality = 3; 
                   if(wd.getSchedulerData() == NULL) { 
                       wd.setSchedulerData((ScheduleWDData*)criticality, true); } 
                   else { 
                        ScheduleWDData *scData = wd.getSchedulerData(); 
                      scData = (ScheduleWDData*)criticality; 
                   }
                   fprintf(stderr, "Botlevel submited wd %d criticality = %d\n", wd.getId(), *((int*)wd.getSchedulerData()));
#endif
                   return;
                }
                 // Add it to the priority queue
                wd.setPriority(priority);
                if( wd.getPriority() >  currMax)//maxBotLev )
                {
                   {
                      LockBlock l(botLevLock);
                      maxBotLev = wd.getPriority();
                      currMax = maxBotLev;
                      topWD = &wd;
                      topSuccesors = (dos->getSuccessors());
                      numCritical++;
                   }
                   qId = 1;
                   NANOS_INSTRUMENT ( *criticality = 1; )
                }
                else if (!strict && (wd.getPriority() ==  currMax) )
                {
                   {
                      LockBlock l(botLevLock);
                      maxBotLev = wd.getPriority();
                      currMax = maxBotLev;
                      topWD = &wd;
                      topSuccesors = (dos->getSuccessors());
                   }
                   qId = 1;
                   NANOS_INSTRUMENT ( *criticality = 1; )
                }
                else
                {
                   //bottom levels are computed dynamically so a successor botlevel may become greater than the botlevel of its predecessor
                   if( ((topSuccesors.find( dos )) != (topSuccesors.end()))   
                      && wd.getPriority() >= currMax-1 )            
                   {
                      //we are in the critical path!
                      {
                         LockBlock l(botLevLock);
                         currMax = wd.getPriority();
                         topWD = &wd; 
                         topSuccesors = (dos->getSuccessors());
                         numCritical++;
                      }
                      qId = 1;
                      // FOR GRAPH REPRESENTATION: 
                      // data._readyQueues[3].push_back( &wd );
                      NANOS_INSTRUMENT ( *criticality = 1; )
                   }
                   else 
                   {
                      //Non-critical task
                      qId = 0;
                      NANOS_INSTRUMENT ( *criticality = 2; )
                   }
                }
                data._readyQueues[qId].push_back( &wd ); //queues 0 or 1
                dodata->setReady();

#ifdef NANOS_INSTRUMENTATION_ENABLED
                   if(wd.getSchedulerData() == NULL) {
                       wd.setSchedulerData((ScheduleWDData*)criticality, true); }
                   else {
                        ScheduleWDData *scData = wd.getSchedulerData();
                      scData = (ScheduleWDData*)criticality;
                   }
                   fprintf(stderr, "Botlevel submited wd %d criticality = %d\n", wd.getId(), *((int*)wd.getSchedulerData()));
      
#endif
                // FOR GRAPH REPRESENTATION:
                //data._readyQueues[3].push_back( &wd );
                return;
            }

            void updateBottomLevels( BotLevDOData *dodata, int botLev )
            {
               std::vector<BotLevDOData *> stack;
               //! Set the bottom level, and add to the stack
               dodata->setBotLevel(botLev);
               stack.push_back(dodata);
               if( dodata->getReady() )  return;
               // A depth first traversal
               while ( !stack.empty() ) {

                  // Pop an element from the stack and get its level
                  BotLevDOData *bd = stack.back();
                  stack.pop_back();

                  int botLevNext = bd->getBotLevel() + 1;
                  bool changed = false;
                  // Deal with the predecessors
                  std::set<BotLevDOData *> *preds = bd->getPredecessors();
                  
                  for ( std::set<BotLevDOData *>::iterator pIter = preds->begin(); pIter != preds->end(); pIter++ )
                  {
                     BotLevDOData *pred = *pIter;
                     if(botLevNext > pred->getBotLevel()) {
                        changed = pred->setBotLevel( botLevNext );
                        stack.push_back( pred );
                        if(changed) { }
                     }
#if SUPPORT_UPDATE
                     // update the priorities to the ready tasks
                     WD * predWD = pred->getWorkDescriptor();
                     if ( changed && pred->getReady() && predWD ) {
                           TeamData &data = ( TeamData & ) *myThread->getTeam()->getScheduleData();
                           int criticality = *((int*)predWD->getSchedulerData());//predWD->getCriticality();
                           if(criticality == 1)
                              data._readyQueues[2].update_priority( predWD, botLevNext);
                           else if(criticality == 2)
                              data._readyQueues[0].update_priority( predWD, botLevNext);
                           else if(criticality == 3)
                              data._readyQueues[3].update_priority( predWD, botLevNext);
                     } 
#endif
                  }
               }
            }


            void atCreate ( DependableObject &depObj )
            {
               NANOS_INSTRUMENT( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
               NANOS_INSTRUMENT( static nanos_event_key_t wd_atCreate  = ID->getEventKey("wd-atCreate"); )
               NANOS_INSTRUMENT( WD * relObj = (WD*)depObj.getRelatedObject(); )
               NANOS_INSTRUMENT( unsigned wd_id = ((WD *)relObj)->getId( ); )
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( wd_atCreate, wd_id ); )
              
               NANOS_INSTRUMENT( static nanos_event_key_t blev_overheads  = ID->getEventKey("blev-overheads"); )
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( blev_overheads, 1 ); ) 
               NANOS_INSTRUMENT( static nanos_event_key_t blev_overheads_br  = ID->getEventKey("blev-overheads-breakdown"); )
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( blev_overheads_br, NANOS_SCHED_BLEV_ATCREATE ); )

               //! Creating DO scheduler data
               BotLevDOData *dodata = new BotLevDOData(++taskNumber, 0);
               depObj.setSchedulerData( (DOSchedulerData*) dodata );
               int iteration = 0;
               //! Duplicating dependence info depObj -> dodata 
               std::set<DependableObject *> predecessors = depObj.getPredecessors();
               for ( std::set<DependableObject *>::iterator it = predecessors.begin(); it != predecessors.end(); it++ ) {
                  iteration++;
                  DependableObject *pred = *it;
                  if (pred) {
                     BotLevDOData *predObj = (BotLevDOData *)pred->getSchedulerData();
                     if (predObj) {
                        dodata->addPredecessor(predObj);
                     }
                  }
               }

               //! When reaching the threshold we need to update bottom levels,
               //! otherwise we push dodata in a temporal stack

               //! \todo FIXME stack needs a thread safe mechanism
#if 1 
               if ( _blStack.size() + 1 >= (unsigned int) updateFreq ) {
                  updateBottomLevels(dodata, 0);
                  while ( !_blStack.empty() ) {
                     BotLevDOData *dodata1 = _blStack.top();
                     if (dodata1->getBotLevel() <= 0) updateBottomLevels(dodata1, 0);
                     _blStack.pop();
                  }
               } else {
                  _blStack.push( dodata );
               }
#else
               if ( _blStack.size() + 1 >= (unsigned int) updateFreq ) {
                  updateBottomLevels(dodata, 0);
               }
#endif
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( wd_atCreate, wd_id ); )
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( blev_overheads, 1 ); )
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( blev_overheads_br, NANOS_SCHED_BLEV_ATCREATE ); )
            }

            /*!
            *  \brief Function called when a new task must be created: the new created task
            *          is directly queued (Breadth-First policy)
            *  \param thread pointer to the thread to which belongs the new task
            *  \param wd a reference to the work descriptor of the new task
            *  \sa WD and BaseThread
            */
            virtual WD * atSubmit ( BaseThread *thread, WD &newWD )
            {
               NANOS_INSTRUMENT( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
               NANOS_INSTRUMENT( static nanos_event_key_t sched_overheads  = ID->getEventKey("sched-overheads"); )
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sched_overheads, 1 ); )
               NANOS_INSTRUMENT( static nanos_event_key_t blev_overheads  = ID->getEventKey("blev-overheads"); )
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( blev_overheads, 1 ); )
               NANOS_INSTRUMENT( static nanos_event_key_t blev_overheads_br  = ID->getEventKey("blev-overheads-breakdown"); )
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( blev_overheads_br, NANOS_SCHED_BLEV_ATSUBMIT ); )

               queue(thread,newWD);

               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sched_overheads, 1 ); )
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( blev_overheads, 1 ); )
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( blev_overheads_br, NANOS_SCHED_BLEV_ATSUBMIT ); )

               return 0;
            }

            virtual WD *atIdle ( BaseThread *thread );
            virtual void atShutdown();
      };

      /*! 
       *  \brief Function called by the scheduler when a thread becomes idle to schedule it: implements the CILK-scheduler algorithm
       *  \param thread pointer to the thread to be scheduled
       *  \sa BaseThread
       */
      WD * BotLev::atIdle ( BaseThread *thread )
      {
         NANOS_INSTRUMENT( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
         NANOS_INSTRUMENT( static nanos_event_key_t sched_overheads  = ID->getEventKey("sched-overheads"); )
         NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( sched_overheads, 1 ); )
         NANOS_INSTRUMENT( static nanos_event_key_t blev_overheads  = ID->getEventKey("blev-overheads"); )
         NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( blev_overheads, 1 ); )
         NANOS_INSTRUMENT( static nanos_event_key_t blev_overheads_br  = ID->getEventKey("blev-overheads-breakdown"); )
         NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( blev_overheads_br, NANOS_SCHED_BLEV_ATIDLE ); )

         WorkDescriptor * wd;
         TeamData &data = ( TeamData & ) *thread->getTeam()->getScheduleData();
         unsigned int spins = numSpins;

         //fprintf(stderr, "Num threads = %u\n",  myThread->getTeamData()->getTeam()->size());
         //Separation of big and small cores - big cores execute from queue 1 - small cores execute from queue 2
         if( ( myThread->getTeamData()->getTeam()->getNumSupportingThreads() > 1) && ((thread->runningOn()->getId() >= hpFrom && thread->runningOn()->getId() <= hpTo) || 
             ( hpSingle && thread->runningOn()->getId() == hpSingle ))) {
            //Big core
            wd = data._readyQueues[1].pop_front( thread );
            while( wd == NULL && spins )
            {
               wd = data._readyQueues[1].pop_front( thread );
               spins--;
            }
            if(!wd ) {
               // default work stealing: big stealing from small
               wd = data._readyQueues[0].pop_front( thread );
            }
         }
         else {
            //Small core
            wd = data._readyQueues[0].pop_front( thread );
            while( wd == NULL && spins )
            {
               wd = data._readyQueues[0].pop_front( thread );
               spins--;
            }
            if(!wd && steal) {
                //optionally: small stealing from big
                wd = data._readyQueues[1].pop_front( thread );
            }
          
         }
         NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( sched_overheads, 1 ); )
         NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( blev_overheads, 1 ); )
         NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( blev_overheads_br, NANOS_SCHED_BLEV_ATIDLE ); )

         if(!wd) wd = data._readyQueues[2].pop_front( thread );   //return wd;
         //return  data._readyQueues[3].pop_front( thread );

         if(wd) {
            NANOS_INSTRUMENT ( static nanos_event_key_t criticalityKey = ID->getEventKey("wd-criticality"); )
            NANOS_INSTRUMENT ( nanos_event_value_t wd_criticality; )
            NANOS_INSTRUMENT ( wd_criticality = *((nanos_event_value_t*)wd->getSchedulerData()); )
            NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenBurstEvent ( criticalityKey, wd_criticality ); )
            NANOS_INSTRUMENT ( fprintf(stderr, "Threw event! criticality = %d\n", (int)wd_criticality); )
         }
         return wd;
      }

      void BotLev::atShutdown() {
         fprintf(stderr, "CRITICAL TASKS: %u\n", numCritical);
         fprintf(stderr, "TOTAL TASKS: %u\n", taskNumber);
      }

      class BotLevSchedPlugin : public Plugin
      {
         public:
            BotLevSchedPlugin() : Plugin( "Distributed Breadth-First scheduling Plugin",1 ) {}

            virtual void config ( Config &config_ )
            {
                config_.setOptionsSection( "Bottom level", "Bottom-level scheduling module" );
                config_.registerConfigOption ( "update-freq", new Config::PositiveVar( updateFreq ), "Defines how often to update the bottom level" );
                config_.registerArgOption ( "update-freq", "update-freq" );
                config_.registerEnvOption ( "update-freq", "NX_BL_FREQ" );

                config_.registerConfigOption ( "hpcores", new Config::PositiveVar( hpCores ), "Defines how many big cores the system has" );
                config_.registerArgOption ( "hpcores", "hpcores" );
                config_.registerEnvOption ( "hpcores", "NX_HP_CORES" );

                config_.registerConfigOption ( "numSpins", new Config::PositiveVar( numSpins ), "Defines the number of spins in atIdle (work stealing)" );
                config_.registerArgOption ( "numSpins", "numSpins" );
                config_.registerEnvOption ( "numSpins", "NX_NUM_SPINS" );

                config_.registerConfigOption ( "from", new Config::PositiveVar( hpFrom ), "Defines the number of spins in atIdle (work stealing)" );
                config_.registerArgOption ( "from", "from" );
                config_.registerEnvOption ( "from", "NX_HP_FROM" );

                config_.registerConfigOption ( "to", new Config::PositiveVar( hpTo ), "Defines the number of spins in atIdle (work stealing)" );
                config_.registerArgOption ( "to", "hpTo" );
                config_.registerEnvOption ( "to", "NX_HP_TO" );

                config_.registerConfigOption ( "single", new Config::IntegerVar( hpSingle ), "Defines the number of spins in atIdle (work stealing)" );
                config_.registerArgOption ( "single", "hpSingle" );

                config_.registerConfigOption ( "maxBlev", new Config::IntegerVar( maxBL ), "Defines the initial value of maximum bottom level" );
                config_.registerArgOption ( "maxBlev", "maxBlev" );
                config_.registerEnvOption ( "maxBlev", "NX_MAXB" );

                config_.registerConfigOption ( "strict", new Config::IntegerVar( strict ), "Defines if we use strict policy or not" );
                config_.registerArgOption ( "strict", "strict" );
                config_.registerEnvOption ( "strict", "NX_STRICTB" );

                config_.registerConfigOption ( "steal", new Config::IntegerVar( steal ), "Defines if we use work stealing big <-- small" );
                config_.registerArgOption ( "steal", "steal" );
                config_.registerEnvOption ( "steal", "NX_STEALB" );
            }

            virtual void init() {
               sys.setDefaultSchedulePolicy(new BotLev());
            }
      };

   }
}

DECLARE_PLUGIN("botlev",nanos::ext::BotLevSchedPlugin);
