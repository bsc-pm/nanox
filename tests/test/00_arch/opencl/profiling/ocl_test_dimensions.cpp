/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
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

/*
<testinfo>
test_generator=gens/opencl-generator
test_generator_ENV='test_architecture=smp'
test_schedule=bf
</testinfo>
*/

#include "assert.h"
#include <iostream>
#include "openclprocessor.hpp"

using namespace std;
using namespace nanos;
using namespace ext;

int main()
{
  // Class to check
  ::OpenCLAdapter *openCLAdapter;
  openCLAdapter = new ::OpenCLAdapter();

  cl_kernel kernel1 = (cl_kernel) 1111;

  ::Execution *execution1, *execution2;
  execution1 = new ::Execution(3,1,2,3,10);
  execution2 = new ::Execution(3,1,2,3,5);

  std::map<cl_kernel,OpenCLAdapter::DimsBest> bestExec;
  std::map<cl_kernel,OpenCLAdapter::DimsExecutions> nExec;

  ::Dims dims1(3,2,4,8,1.0);
  ::Dims dims2(3,2,2,8,1.0);

  ::Dims dims3(3,2,6,8,1.0);
  ::Dims dims4(3,2,4,7,1.0);

  ::OpenCLAdapter::DimsBest kernelDims;

  /* Begin testing */

  // Several dimensions
  assert(openCLAdapter->getBestExec().size() == 0);

  openCLAdapter->updateProfiling(kernel1, execution1/* 10 */, dims1); // Add one execution
  assert(openCLAdapter->getBestExec().size() == 1);
  bestExec = openCLAdapter->getBestExec();
  nExec = openCLAdapter->getExecutions();
  kernelDims = bestExec[kernel1];
  assert(kernelDims.size() == 1);
  assert(kernelDims[dims1]->getTime() == 10);

  openCLAdapter->updateProfiling(kernel1, execution2/* 5 */, dims2); // Add one execution
  assert(openCLAdapter->getBestExec().size() == 1);
  bestExec = openCLAdapter->getBestExec();
  nExec = openCLAdapter->getExecutions();
  kernelDims = bestExec[kernel1];
  assert(kernelDims.size() == 2);
  assert(kernelDims[dims1]->getTime() == 10);
  assert(kernelDims[dims2]->getTime() == 5);

  openCLAdapter->updateProfiling(kernel1, execution2/* 5 */, dims3); // Add one execution
  assert(openCLAdapter->getBestExec().size() == 1);
  bestExec = openCLAdapter->getBestExec();
  nExec = openCLAdapter->getExecutions();
  kernelDims = bestExec[kernel1];
  assert(kernelDims.size() == 3);
  assert(kernelDims[dims1]->getTime() == 10);
  assert(kernelDims[dims2]->getTime() == 5);
  assert(kernelDims[dims3]->getTime() == 5);

  openCLAdapter->updateProfiling(kernel1, execution1/* 10 */, dims4); // Add one execution
  assert(openCLAdapter->getBestExec().size() == 1);
  bestExec = openCLAdapter->getBestExec();
  nExec = openCLAdapter->getExecutions();
  kernelDims = bestExec[kernel1];
  assert(kernelDims.size() == 4);
  assert(kernelDims[dims1]->getTime() == 10);
  assert(kernelDims[dims2]->getTime() == 5);
  assert(kernelDims[dims3]->getTime() == 5);
  assert(kernelDims[dims4]->getTime() == 10);

  /* End testing */

  delete execution1;
  delete execution2;

  return 0;
}
