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

#include "smpprocessor.hpp"
#include "schedule.hpp"
#include "debug.hpp"
#include "system.hpp"
#include <iostream>

extern "C"
{
// low-level routine to switch stacks
   void switchStacks( void *,void *,void *,void * );
}

using namespace nanos;

void * smp_bootthread ( void *arg )
{
   SMPThread *self = static_cast<SMPThread *>( arg );

   self->run();

   pthread_exit ( 0 );
}

void SMPThread::start ()
{
// TODO:
//        /* initialize thread_attr: init. attr */
//        pthread_attr_init(&nth_data.thread_attr);
//        /* initialize thread_attr: stack attr */
//        rv_pthread = pthread_attr_setstack(
//                         (pthread_attr_t *) &nth_data.thread_attr,
//                         (void *) aux_desc->stack_addr,
//                         (size_t) aux_desc->stack_size
//                 );


   if ( pthread_create( &pth, NULL, smp_bootthread, this ) )
      fatal( "couldn't create thread" );
}

void SMPThread::runDependent ()
{
   WD &work = getThreadWD();
   setCurrentWD( work );

   SMPDD &dd = ( SMPDD & ) work.activateDevice( SMP );
   dd.getWorkFct()( work.getData() );
}

void SMPThread::join ()
{
   pthread_join( pth,NULL );
}

// This is executed in between switching stacks
static void switchHelper ( WD *oldWD, WD *newWD, intptr_t *oldState  )
{
   SMPDD & dd = ( SMPDD & )oldWD->getActiveDevice();
   dd.setState( oldState );
   Scheduler::queue( *oldWD );
   myThread->setCurrentWD( *newWD );
}

void SMPThread::inlineWork ( WD *wd )
{
   SMPDD &dd = ( SMPDD & )wd->getActiveDevice();
   WD *oldwd = getCurrentWD();
   setCurrentWD( *wd );
   ( dd.getWorkFct() )( wd->getData() );
   // TODO: not delete work descriptor if is a parent with pending children
   setCurrentWD( *oldwd );
}

void SMPThread::switchTo ( WD *wd )
{
   // wd MUST have an active Device when it gets here
   ensure( wd->hasActiveDevice(),"WD has no active SMP device" );
   SMPDD &dd = ( SMPDD & )wd->getActiveDevice();

   if ( useUserThreads ) {
      debug( "switching from task " << getCurrentWD() << ":" << getCurrentWD()->getId() << " to " << wd << ":" << wd->getId() );

      if ( !dd.hasStack() ) {
         dd.initStack( wd->getData() );
      }

      ::switchStacks(

         ( void * ) getCurrentWD(),
         ( void * ) wd,
         ( void * ) dd.getState(),
         ( void * ) switchHelper );
   } else {
      inlineWork( wd );
      delete wd;
   }
}

static void exitHelper (  WD *oldWD, WD *newWD, intptr_t *oldState )
{
   delete oldWD;
   myThread->setCurrentWD( *newWD );
}

void SMPThread::exitTo ( WD *wd )
{
   // wd MUST have an active Device when it gets here
   ensure( wd->hasActiveDevice(),"WD has no active SMP device" );
   SMPDD &dd = ( SMPDD & )wd->getActiveDevice();

   debug( "exiting task " << getCurrentWD() << ":" << getCurrentWD()->getId() << " to " << wd << ":" << wd->getId() );
   // TODO: reuse stack

   if ( !dd.hasStack() ) {
      dd.initStack( wd->getData() );
   }

   //TODO: optimize... we don't really need to save a context in this case
   ::switchStacks(
      ( void * ) getCurrentWD(),
      ( void * ) wd,
      ( void * ) dd.getState(),
      ( void * ) exitHelper );
}

void SMPThread::bind( void )
{
   cpu_set_t cpu_set;
   int cpu_id = getCpuId();

   ensure( ( ( cpu_id >= 0 ) && ( cpu_id < CPU_SETSIZE ) ), "invalid value for cpu id" );
   CPU_ZERO( &cpu_set );
   CPU_SET( cpu_id, &cpu_set );
   verbose( "Binding thread " << getId() << " to cpu " << cpu_id );
   sched_setaffinity( ( pid_t ) 0, sizeof( cpu_set ), &cpu_set );
}

