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

/*
<testinfo>
test_generator=gens/opencl-generator
test_generator_ENV=( "NX_TEST_SCHEDULE=bf"
                     "NX_TEST_ARCH=smp" )
</testinfo>
*/

#include "assert.h"
#include <iostream>
#include <map>
#include "openclprocessor.hpp"
#include "openclprofiler.hpp"

using namespace std;
using namespace nanos;
using namespace ext;

int main()
{
  // Class to check
  ::OpenCLAdapter *openCLAdapter;
  openCLAdapter = new ::OpenCLAdapter();

  std::string kernel1 = "kernel1";
  std::string kernel2 = "kernel2";

  Execution execution1(3,1,2,3,10,0,0,0,false,false);
  Execution execution2(3,1,2,3,5,0,0,0,false,false);

  std::map<std::string,DimsBest> bestExec;
  std::map<std::string,DimsExecutions> nExec;

  ::Dims dims1(3,2,4,8,1.0);
  ::Dims dims2(3,8,4,2,1.0);

  DimsBest dims1Best;

  /* Begin testing */

  assert(openCLAdapter->getBestExec().size() == 0);
  // Several kernels
  openCLAdapter->updateProfStats(kernel1, dims1, execution1/* 10 */); // Add one execution
  assert(openCLAdapter->getBestExec().size() == 1);
  bestExec = openCLAdapter->getBestExec();
  nExec = openCLAdapter->getExecutions();
  dims1Best = bestExec[kernel1];
  assert(dims1Best.size() == 1);
  assert(dims1Best[dims1].getTime() == 10);

  openCLAdapter->updateProfStats(kernel2, dims2, execution2/* 5 */); // Add one execution
  assert(openCLAdapter->getBestExec().size() == 2);
  bestExec = openCLAdapter->getBestExec();
  dims1Best = bestExec[kernel2];
  assert(dims1Best[dims2].getTime() == 5);

  /* End testing */

  // delete openCLAdapter;
  /*
   * OpenCLAdapter was not initialized, thus the destructor will try to release memory not allocated
   */

  return 0;
}

// Enable OpenCL automatic detection
__attribute__((weak)) void nanos_needs_opencl_fun(void) {}
