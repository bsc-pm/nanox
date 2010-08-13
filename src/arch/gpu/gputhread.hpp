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

#ifndef _NANOS_GPU_THREAD
#define _NANOS_GPU_THREAD

#include "gpudd.hpp"
#include "smpthread.hpp"


namespace nanos {
namespace ext
{

   class GPUThread : public SMPThread
   {

      friend class GPUProcessor;

      class PendingCopy : nanos::WG {

         private:
            void *                  _src;
            void *                  _dst;
            size_t                  _size;
            DependableObject        _do;
            DependenciesDomain &    _dd;


         public:
            PendingCopy( void * source, void * dest, size_t s ) :
                  _src( source ), _dst( dest ), _size( s ), _do(),
                  _dd( myThread->getCurrentWD()->getParent()->getDependenciesDomain() )
            {
               //_dd = myThread->getCurrentWD()->getParent()->getDependenciesDomain();
               std::vector<Dependency> dependencies;
               dependencies.push_back( Dependency( &_dst, 0, false, true ) );
               _dd.submitDependableObject( _do, dependencies );
               //myThread->getCurrentWD()->getParent()->addWork( ( WG )this );
            }

            ~PendingCopy()
            {
               _dd.finished( _do );
               WG::done();
            }

            void * getSrc ()
            {
               return _src;
            }

            void * getDst ()
            {
               return _dst;
            }

            size_t getSize ()
            {
               return _size;
            }

            const PendingCopy& operator=(const PendingCopy& pd)
            {
               _src = pd._src;
               _dst = pd._dst;
               _size = pd._size;
               _do = pd._do;
               //_dd( myThread->getCurrentWD()->getParent()->getDependenciesDomain() );

               return *this;
            }

            void executeAsyncCopy();
            void executeSyncCopy();

      };

      private:
         int                        _gpuDevice; // Assigned GPU device Id
         std::vector<PendingCopy>   _pendingCopies;

         // disable copy constructor and assignment operator
         GPUThread( const GPUThread &th );
         const GPUThread & operator= ( const GPUThread &th );

      public:
         // constructor
         GPUThread( WD &w, PE *pe, int device ) : SMPThread( w, pe ), _gpuDevice( device ), _pendingCopies() {}

         // destructor
         virtual ~GPUThread() {}

         virtual void runDependent ( void );

         virtual void inlineWorkDependent( WD &work );

         /** \brief GPU specific yield implementation
         */
         virtual void yield();

         int getGpuDevice ()
         {
            return _gpuDevice;
         }

         void addPendingCopy ( void * source, void * dest, size_t size )
         {
            _pendingCopies.push_back( PendingCopy( source, dest, size ) );
            myThread->getCurrentWD()->getParent()->addWork( (WG &) _pendingCopies.back() );
         }

   };


}
}

#endif
