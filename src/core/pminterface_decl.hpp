/*************************************************************************************/
/*      Copyright 2010 Barcelona Supercomputing Center                               */
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

#ifndef NANOS_PM_INTERFACE
#define NANOS_PM_INTERFACE

#include "config.hpp"

class PMInterface
{
   private:
      /*! \brief PMInterface copy constructor (private)
       */
      PMInterface( PMInterface &pmi );
      /*! \brief PMInterface copy assignment operator (private)
       */
      PMInterface& operator= ( PMInterface &pmi );
   public:
      /*! \brief PMInterface default constructor
       */
      PMInterface() {}
      /*! \brief PMInterface destructor
       */
      virtual ~PMInterface() {}

      virtual int getInternalDataSize() const { return 0; }

      virtual void config (Config &cfg) {}
      virtual void start () {}
      virtual void finish() {}

      virtual void setupWD( WD &wd ) {}
      virtual void wdStarted( WD &wd ) {}
      virtual void wdFinished( WD &wd ) {}
};

#endif /* PM_INTERFACE_HPP_ */
