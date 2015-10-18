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

#include "openclprofiler.hpp"
#include <string>
#include <sstream>
#include "debug.hpp"
#include "openclutils.hpp"

using namespace nanos;

OpenCLProfilerException::OpenCLProfilerException(nanos::OpenCLProfilerExceptions exception, cl_int clError, char* errorString)
{
   std::string baseMsg = "[OpenCL Profiling Error] : ";
   std::stringstream errorBuffer;
   switch ( exception ) {
      case CLP_WRONG_NUMBER_OF_DIMENSIONS:
         fatal0( baseMsg + "Wrong number of dimensions");
         break;
      case CLP_OPENCL_STANDARD_ERROR:
         errorBuffer << clError;
         fatal0( baseMsg + errorBuffer.str() + ", " + errorString);
         break;
      default:
         fatal0( baseMsg + "Generic" );
         break;
   }
}

uint32_t OpenCLProfilerDbManager::getKernelHash(const Dims &dims, const std::string kernelName)
{
  // Hash generation based on kernel name and dimensions
  std::stringstream fullString;
  fullString << kernelName;
  switch ( dims.getNdims() ) {
  case 1:
    fullString << dims.getGlobalX();
    break;
  case 2:
    fullString << dims.getGlobalX()+dims.getGlobalY();
    break;
  case 3:
    fullString << dims.getGlobalX()+dims.getGlobalY()+dims.getGlobalZ();
    break;
  default:
    break;
  }
  return nanos::ext::gnuHash(fullString.str().c_str());
}

/**
 * @brief callback for the SQL select statement execution
 */
int oclDbSelectCallback(void *Execution, int argc, char **argv, char **azColName);

/**
 * @brief callback for the SQL insert execution
 */
int oclDbInsertCallback(void *notUsed, int argc, char **argv, char **azColName);

/**
 * @brief callback for the sql statement execution
 */
int oclDbCheckCallback(void *nrows, int argc, char **argv, char **azColName);

int oclDbInsertCallback(void *notUsed, int argc, char **argv, char **azColName)
{
  return 0;
}

int oclDbSelectCallback(void *openCLProfilerDbManager, int argc, char **argv, char **azColName)
{
  if ( argc != 3 )
    throw; // Select statement expect only three columns: x, y and z

  nanos::Execution *bestExecution;
  bestExecution = new nanos::Execution(0/*Not read*/, atoi(argv[0]), atoi(argv[1]), atoi(argv[2]), 0/*Not read*/);

  nanos::OpenCLProfilerDbManager *openCLProfilerDbManagerPtr = static_cast<nanos::OpenCLProfilerDbManager*>(openCLProfilerDbManager);

  openCLProfilerDbManagerPtr->setExecution(bestExecution);

  // TODO: Delete this
  for ( int i=0; i<argc; i++ )
  {
     printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");

  }

  return 0;
}


int oclDbCheckCallback(void *nrows, int argc, char **argv, char **azColName)
{
  *((int*)nrows) = argc; // Set the number of columns to check if exists
  return 0;
}

void OpenCLProfilerDbManager::setKernelConfig(Dims &dims, Execution &execution, std::string kernelName)
{
  const std::string tableString = "CREATE TABLE IF NOT EXISTS opencl_kernels(hash INT, dims INT, x INT, y INT, z INT, PRIMARY KEY(hash));";
  std::stringstream finalStmt; // Schema: 'INSERT INTO TABLE VALUES(HASH, DIMS, X, Y, Z)
  if ( !isTableCreated() )
    finalStmt << tableString;

  // SQL statement generation
  const std::string baseInsertStmt = "INSERT INTO " + CL_PROFILING_DEFAULT_TABLE + " VALUES(";
  finalStmt << baseInsertStmt;
  finalStmt << getKernelHash(dims, kernelName) << ",";
  finalStmt << static_cast<int>(dims.getNdims()) << ",";
  finalStmt << execution.getLocalX() << ",";
  finalStmt << execution.getLocalY() << ",";
  finalStmt << execution.getLocalZ();
  finalStmt << ")";

  _dbManager.executeStatement(finalStmt.str(), oclDbInsertCallback);
  _created = true;
}

Execution* OpenCLProfilerDbManager::getKernelConfig(Dims &dims, std::string kernelName)
{
  if ( isTableCreated() ) {
    destroyExecution();
    std::stringstream selectStmt;
    selectStmt << "SELECT x,y,z FROM ";
    selectStmt << CL_PROFILING_DEFAULT_TABLE;
    selectStmt << " WHERE hash = ";
    selectStmt << getKernelHash(dims, kernelName);
    _dbManager.executeStatement(selectStmt.str(), oclDbSelectCallback, this);
    if ( !_isExecutionSet )
      setExecution(new Execution(9,0,0,0,0)); // No configurations found
  } else {
    setExecution(new Execution(10,0,0,0,0));   // No table found
  }
  return _execution;
}

bool OpenCLProfilerDbManager::isTableCreated()
{
  if ( !_created ) {
    int nrows = 0; // TODO: Optimize this query!!!!!
    const std::string checkTableStmt = "PRAGMA table_info(" + CL_PROFILING_DEFAULT_TABLE + ")";
    _dbManager.executeStatement(checkTableStmt, oclDbCheckCallback, (void*)&nrows);
    _created = nrows > 0 ? true : false;
  }
  return _created;
}

bool OpenCLProfilerDbManager::createTable()
{
  const std::string tableString = "CREATE TABLE IF NOT EXISTS opencl_kernels(hash INT, dims INT, x INT, y INT, z INT, PRIMARY KEY(hash));";
  return _dbManager.executeStatement(tableString, oclDbSelectCallback);
}

OpenCLProfilerDbManager::~OpenCLProfilerDbManager()
{
  destroyExecution();
}

void OpenCLProfilerDbManager::destroyExecution()
{
  if ( _isExecutionSet ) {
    delete _execution;
    _isExecutionSet = false;
    _execution = NULL;
  }
}
