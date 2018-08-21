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

#include "schedule.hpp"
#include "wddeque.hpp"
#include "plugin.hpp"
#include "system.hpp"

#include <iostream>
#include <fstream>
#include <assert.h>

#define SUPPORT_UPDATE 1 

namespace nanos {
   namespace ext {
      /*! \brief Configuration options for the BotLev scheduling policy
       *  are stored here instead of in the namespace.
       */
      struct BotLevCfg
      {
         public:
            static int   updateFreq;
            static int   numSpins;
            static int   hpFrom;
            static int   hpTo;
            static int   hpSingle;
            static int   steal;
            static int   maxBL;
            static int   strict;
            static int   taskNumber;
            NANOS_INSTRUMENT( static int   numCritical; )   //! The number of critical tasks (for instrumentation)
      };
      
      // Initialise default values
      int BotLevCfg::updateFreq = 0;
      int BotLevCfg::numSpins = 300;
      int BotLevCfg::hpFrom = 0;
      int BotLevCfg::hpTo = 0;
      int BotLevCfg::hpSingle = 0;
      int BotLevCfg::steal = 0;
      int BotLevCfg::maxBL = 1;
      int BotLevCfg::strict = 0;
      int BotLevCfg::taskNumber = 0;
      NANOS_INSTRUMENT( int BotLevCfg::numCritical; )

      /* Helper class for the computation of bottom levels 
         One instance of this class is stored inside each dependableObject.
      */
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
            predecessors_t    _predecessors;  //! List of BotLevDOData predecessors 
            short             _isCritical;    //! Is the task critical -- needed for reordering the ready queues

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
 
            void setCriticality( short c ) { _isCritical = c; }
            short getCriticality()         { return _isCritical; }

            WD* getWorkDescriptor() const { return _wd; }
            void setWorkDescriptor(WD *wd) { _wd = wd; }

            std::set<BotLevDOData *> *getPredecessors() { return &_predecessors; }
            void addPredecessor(BotLevDOData *predDod) { _predecessors.insert(predDod); }
      };

      class BotLev : public SchedulePolicy
      {
         public:
            using SchedulePolicy::queue;
            typedef std::stack<BotLevDOData *>   bot_lev_dos_t;
            typedef std::set<std::pair< unsigned int, DependableObject * > > DepObjVector; /**< Type vector of successors  */

         private:
            bot_lev_dos_t     _blStack;       //! tasks added, pending having their bottom level updated
            Lock              _stackLock;
            DepObjVector      _topSuccesors;  //! Successors of the last maxPriority task
            Lock              _botLevLock;    //! Lock used for topSuccessors and currMax
            int               _currMax;       //! The priority of the last critical task
            int               _maxBotLev;     //! The maximum priority of the tdg

            struct TeamData : public ScheduleTeamData
            {
               /*! queues of ready tasks to be executed */
               WDPriorityQueue<> *_readyQueues;
               TeamData () : ScheduleTeamData()
               {
                  _readyQueues = NEW WDPriorityQueue<>[3];
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
               sys.setPredecessorLists(true);
               _currMax = _maxBotLev = BotLevCfg::maxBL;
               NANOS_INSTRUMENT( BotLevCfg::numCritical = 0; )
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

            struct WDData : public ScheduleWDData
            {
               int _criticality;

               void setCriticality( int cr ) { _criticality = cr; }
               int getCriticality ( ) { return _criticality; }
               WDData () : _criticality( 0 ) {}
               virtual ~WDData() {}
            };


            /*!
            *  \brief Enqueue a work descriptor in the readyQueue of the passed thread
            *  \param thread pointer to the thread to which readyQueue the task must be appended
            *  \param wd a reference to the work descriptor to be enqueued
            *  \sa ThreadData, WD and BaseThread
            */
            virtual void queue ( BaseThread *thread, WD &wd )
            {
#ifdef NANOS_INSTRUMENTATION_ENABLED
                int criticality; 
#endif
                TeamData &data = ( TeamData & ) *thread->getTeam()->getScheduleData();
                // Find the priority
                DependableObject *dos = wd.getDOSubmit();
                BotLevDOData *dodata;
                unsigned int priority = 0; 
                short qId = -1;
                if ( dos ){ //&& numThreads>1 ) {
                   dodata = (BotLevDOData *)dos->getSchedulerData();
                   dodata->setWorkDescriptor(&wd);
                   priority = dodata->getBotLevel();
                }
                else {
	           if(wd.getDepth() == 0 ) { 
                      wd.setPriority(0);
                      data._readyQueues[2].push_back( &wd ); //readyQueue number 2 is for the main and implicit tasks
                   }
                   else {  //in this case numThreads = 1 inserting in queue number 2
                      data._readyQueues[2].push_back( &wd );
                   }
#ifdef NANOS_INSTRUMENTATION_ENABLED
                   criticality = 3; 
                   if(wd.getSchedulerData() == NULL) { 
                       WDData * wddata = new WDData();
                       wddata->setCriticality(criticality); 
                       wd.setSchedulerData((ScheduleWDData*)wddata, true);
                   } 
                   else { 
                      WDData & scData = *dynamic_cast<WDData*>( wd.getSchedulerData() ); 
                      scData.setCriticality(criticality); 
                   }
#endif
                   return;
                }
                wd.setPriority(priority);
                /* Critical tasks' consideration
                   1st case: Detection of a new longest path (wdPriority > maxPriority)
                   2nd case: Detection of the next critical task in the current lontest path (belongs in _topSuccessors)
                   3rd case: The remaining tasks are not critical
                */
                if( ( wd.getPriority() >  _maxBotLev ) || 
                    ( !BotLevCfg::strict && (wd.getPriority() ==  _maxBotLev)) ) {
                   //The task is critical
                   {
                      LockBlock l(_botLevLock);
                      _maxBotLev = wd.getPriority();
                      _currMax = _maxBotLev;
                      _topSuccesors = (dos->getSuccessors());
                      NANOS_INSTRUMENT( BotLevCfg::numCritical++; )
                   }
                   dodata->setCriticality(1);
                   qId = 1;
                   NANOS_INSTRUMENT ( criticality = 1; )
                }
                else if( ((_topSuccesors.find( std::make_pair( wd.getId(), dos ) )) != (_topSuccesors.end()))
                         && wd.getPriority() >= _currMax-1 ) {
                   //The task is critical
                   {
                      LockBlock l(_botLevLock);
                      _currMax = wd.getPriority();
                      _topSuccesors = (dos->getSuccessors());
                      NANOS_INSTRUMENT( BotLevCfg::numCritical++; )
                   }
                   dodata->setCriticality(1);
                   qId = 1;
                   NANOS_INSTRUMENT ( criticality = 1; )
                }
                else
                {
                   //Non-critical task
                   dodata->setCriticality(2);
                   qId = 0;
                   NANOS_INSTRUMENT ( criticality = 2; )
                }
                data._readyQueues[qId].push_back( &wd ); //queues 0 or 1
                dodata->setReady();

#ifdef NANOS_INSTRUMENTATION_ENABLED
                if(wd.getSchedulerData() == NULL) {
                       WDData * wddata = new WDData();
                       wddata->setCriticality(criticality);
                       wd.setSchedulerData((ScheduleWDData*)wddata, true);

                }
                else {
                   WDData & scData = *dynamic_cast<WDData*>( wd.getSchedulerData() );
                   scData.setCriticality(criticality);
                }
                WDData & wddata = *dynamic_cast<WDData*>( wd.getSchedulerData() );
                if(wddata.getCriticality() == 1) {
                   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
                   NANOS_INSTRUMENT ( static nanos_event_key_t critical_wd_id = ID->getEventKey("critical-wd-id"); )
                   NANOS_INSTRUMENT ( nanos_event_key_t crit_key[1]; )
                   NANOS_INSTRUMENT ( nanos_event_value_t crit_value[1]; )
                   NANOS_INSTRUMENT ( crit_key[0] = critical_wd_id; )
                   NANOS_INSTRUMENT ( crit_value[0] = (nanos_event_value_t) wd.getId(); )
                   NANOS_INSTRUMENT( sys.getInstrumentation()->raisePointEvents(1, crit_key, crit_value); )
                }
      
#endif
                return;
            }

            void updateBottomLevels( BotLevDOData *dodata, int botLev )
            {
               std::vector<BotLevDOData *> stack;
               //! Set the bottom level, and add to the stack
               dodata->setBotLevel(botLev);
               stack.push_back(dodata);
               //Task is ready so, there are no predecessors to be updated
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
                  
                  for ( std::set<BotLevDOData *>::iterator pIter = preds->begin(); pIter != preds->end(); pIter++ ) {
                     BotLevDOData *pred = *pIter;
                     if(botLevNext > pred->getBotLevel()) {
                        changed = pred->setBotLevel( botLevNext );
                        stack.push_back( pred );
                     }
#if SUPPORT_UPDATE
                     //Reorder the readyQueues if the work descriptor with updated priority is ready
                     WD * predWD = pred->getWorkDescriptor();
                     if ( changed && pred->getReady() && predWD ) {
                           TeamData &data = ( TeamData & ) *myThread->getTeam()->getScheduleData();
                           short criticality = pred->getCriticality();
                           if(criticality == 1)
                              data._readyQueues[1].reorderWD(predWD);
                           else if(criticality == 2)
                              data._readyQueues[0].reorderWD(predWD);
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
               BotLevDOData *dodata = new BotLevDOData(++BotLevCfg::taskNumber, 0);
               depObj.setSchedulerData( (DOSchedulerData*) dodata );

               DepObjVector predecessors;
               { 
                  LockBlock l(depObj.getLock());
                  predecessors = depObj.getPredecessors();
               }
               for ( DepObjVector::iterator it = predecessors.begin(); it != predecessors.end(); it++ ) {
                  DependableObject *pred = it->second;
                  if (pred) {
                     BotLevDOData *predObj = (BotLevDOData *)pred->getSchedulerData();
                     if (predObj) {
                        dodata->addPredecessor(predObj);
                     }
                  }
               }

               //! When reaching the threshold we need to update bottom levels,
               //! otherwise we push dodata in a temporal stack
               if ( _blStack.size() + 1 >= (unsigned int) BotLevCfg::updateFreq ) {
                  updateBottomLevels(dodata, 0);
                  while ( !_blStack.empty() ) {
                     BotLevDOData *dodata1 = _blStack.top();
                     if (dodata1->getBotLevel() <= 0) 
                        updateBottomLevels(dodata1, 0);
                     {
                         LockBlock l(_stackLock);
                         _blStack.pop();
                     }
                  }
               } else {
                  LockBlock l(_stackLock);
                  _blStack.push( dodata );
               }
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
               NANOS_INSTRUMENT( static nanos_event_key_t blev_overheads  = ID->getEventKey("blev-overheads"); )
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( blev_overheads, 1 ); )
               NANOS_INSTRUMENT( static nanos_event_key_t blev_overheads_br  = ID->getEventKey("blev-overheads-breakdown"); )
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( blev_overheads_br, NANOS_SCHED_BLEV_ATSUBMIT ); )

               queue(thread,newWD);

               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( blev_overheads, 1 ); )
               NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( blev_overheads_br, NANOS_SCHED_BLEV_ATSUBMIT ); )

               return 0;
            }
           // virtual void atSuccessor( DependableObject &depObj, DependableObject *pred, atSuccessorFlag mode, int numPred );
            virtual WD *atIdle( BaseThread *thread, int numSteal );
#ifdef NANOS_INSTRUMENTATION_ENABLED
            virtual void atShutdown( );
            virtual WD *atBeforeExit( BaseThread *thread, WD &current, bool schedule );
#endif
      };


/*      void BotLev::atSuccessor( DependableObject &depObj, DependableObject *pred, atSuccessorFlag mode, int numPred )
      {
         if(mode == ADD)
            depObj.addPredecessor(*pred);
         else if(mode == REMOVE_IN_LOCK){ //lock and remove predecessor 
    
            SyncLockBlock lock( depObj.getLock() );

            depObj.decreasePredecessorsInLock( pred, numPred );
         }
         else if(mode == REMOVE) {  //remove without locking
            depObj.decreasePredecessorsInLock( pred, numPred );
         }
         return;
      }
*/
      /*! 
       *  \brief Function called by the scheduler when a thread becomes idle to schedule it: implements the CILK-scheduler algorithm
       *  \param thread pointer to the thread to be scheduled
       *  \sa BaseThread
       */
      WD * BotLev::atIdle ( BaseThread *thread, int numSteal )
      {
         NANOS_INSTRUMENT( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
         NANOS_INSTRUMENT( static nanos_event_key_t blev_overheads  = ID->getEventKey("blev-overheads"); )
         NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( blev_overheads, 1 ); )
         NANOS_INSTRUMENT( static nanos_event_key_t blev_overheads_br  = ID->getEventKey("blev-overheads-breakdown"); )
         NANOS_INSTRUMENT( sys.getInstrumentation()->raiseOpenBurstEvent ( blev_overheads_br, NANOS_SCHED_BLEV_ATIDLE ); )

         WorkDescriptor * wd;
         TeamData &data = ( TeamData & ) *thread->getTeam()->getScheduleData();
         unsigned int spins = BotLevCfg::numSpins;

         //Separation of big and small cores - big cores execute from queue 1 - small cores execute from queue 2
         if( ((thread->runningOn()->getId() >= BotLevCfg::hpFrom && thread->runningOn()->getId() <= BotLevCfg::hpTo) || 
                ( BotLevCfg::hpSingle && thread->runningOn()->getId() == BotLevCfg::hpSingle )) ) {
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
            if(!wd && BotLevCfg::steal) {
                //optionally: small stealing from big
                wd = data._readyQueues[1].pop_front( thread );
            }
          
         }
         NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( blev_overheads, 1 ); )
         NANOS_INSTRUMENT( sys.getInstrumentation()->raiseCloseBurstEvent ( blev_overheads_br, NANOS_SCHED_BLEV_ATIDLE ); )

         if(!wd) wd = data._readyQueues[2].pop_front( thread ); 

#ifdef NANOS_INSTRUMENTATION_ENABLED
         if(wd) {
            NANOS_INSTRUMENT ( static nanos_event_key_t criticalityKey = ID->getEventKey("wd-criticality"); )
            NANOS_INSTRUMENT ( nanos_event_value_t wd_criticality; )
            NANOS_INSTRUMENT ( WDData & wddata = *dynamic_cast<WDData*>( wd->getSchedulerData() ); )
            NANOS_INSTRUMENT ( wd_criticality = wddata.getCriticality(); )
            NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenBurstEvent ( criticalityKey, wd_criticality ); )
         }
#endif
         return wd;
      }

#ifdef NANOS_INSTRUMENTATION_ENABLED
      void BotLev::atShutdown() {
         fprintf(stderr, "\n\nTOTAL TASKS: %u\n", BotLevCfg::taskNumber);
         fprintf(stderr, "CRITICAL TASKS: %u\n", BotLevCfg::numCritical);
         fprintf(stderr, "PERCENTAGE OF CRITICAL TASKS: %f %%\n", (BotLevCfg::numCritical*100)/(double)BotLevCfg::taskNumber);
      }

      WD * BotLev::atBeforeExit(BaseThread *thread, WD &wd, bool schedule) {
         NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
         NANOS_INSTRUMENT ( static nanos_event_key_t criticalityKey = ID->getEventKey("wd-criticality"); )
         NANOS_INSTRUMENT ( nanos_event_value_t wd_criticality; )
         NANOS_INSTRUMENT ( WDData & wddata = *dynamic_cast<WDData*>( wd.getSchedulerData() ); )
         NANOS_INSTRUMENT ( wd_criticality = wddata.getCriticality(); )
         NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseBurstEvent ( criticalityKey, wd_criticality ); )
         return NULL;
      }

#endif
      class BotLevSchedPlugin : public Plugin
      {
         public:
            BotLevSchedPlugin() : Plugin( "Distributed Breadth-First scheduling Plugin",1 ) {}

            virtual void config ( Config &config_ )
            {
                config_.setOptionsSection( "Bottom level", "Bottom-level scheduling module" );
                config_.registerConfigOption ( "update-freq", new Config::PositiveVar( BotLevCfg::updateFreq ), "Defines how often to update the bottom levels" );
                config_.registerArgOption ( "update-freq", "update-freq" );
                config_.registerEnvOption ( "update-freq", "NX_BL_FREQ" );

                config_.registerConfigOption ( "numSpins", new Config::PositiveVar( BotLevCfg::numSpins ), "Defines the number of spins in atIdle (work stealing)" );
                config_.registerArgOption ( "numSpins", "numSpins" );
                config_.registerEnvOption ( "numSpins", "NX_NUM_SPINS" );

                config_.registerConfigOption ( "from", new Config::PositiveVar( BotLevCfg::hpFrom ), "Sets the thread id of the first fast core" );
                config_.registerArgOption ( "from", "from" );
                config_.registerEnvOption ( "from", "NX_HP_FROM" );

                config_.registerConfigOption ( "to", new Config::PositiveVar( BotLevCfg::hpTo ), "Sets the thread id of the last fast core" );
                config_.registerArgOption ( "to", "hpTo" );
                config_.registerEnvOption ( "to", "NX_HP_TO" );

                config_.registerConfigOption ( "single", new Config::IntegerVar( BotLevCfg::hpSingle ), "Sets the thread id of a single fast core" );
                config_.registerArgOption ( "single", "hpSingle" );

                config_.registerConfigOption ( "maxBlev", new Config::IntegerVar( BotLevCfg::maxBL ), "Defines the initial value of maximum bottom level" );
                config_.registerArgOption ( "maxBlev", "maxBlev" );
                config_.registerEnvOption ( "maxBlev", "NX_MAXB" );

                config_.registerConfigOption ( "strict", new Config::IntegerVar( BotLevCfg::strict ), "Defines whether we use strict policy. (Strict -- les crit. tasks, Flexible -- more crit. tasks)" );
                config_.registerArgOption ( "strict", "strict" );
                config_.registerEnvOption ( "strict", "NX_STRICTB" );

                config_.registerConfigOption ( "steal", new Config::IntegerVar( BotLevCfg::steal ), "Defines if we use bi-directional work stealing fast <--> slow" );
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
