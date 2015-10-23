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

#include "dbmanager.hpp"
#include "debug.hpp"
#include "os.hpp"
#include "config.hpp"

using namespace nanos;

DbManager::~DbManager()
{
  cleanPreparedStmts();
  closeConnection();
}

bool DbManager::openConnection(const std::string &databaseName)
{
  if ( _isOpen )
    closeConnection();

  sqlCheck(sqlite3_open(databaseName.c_str(), &_db), "Can't open database: ");

  _isOpen = true;
  return true;
}

void DbManager::cleanPreparedStmts()
{
  for(std::vector<sqlite3_stmt*>::const_iterator iter = _stmtVector.begin(); iter != _stmtVector.end(); iter++)
  {
    sqlCheck(sqlite3_finalize(*iter), "Can't finalize prepared statement");
  }
}

bool DbManager::openConnection()
{
  return openConnection(defaultDbName);
}

void DbManager::sqlCheck(const int err, const std::string &msg)
{
  if ( err != SQLITE_OK ) {
    std::string errMsg = msg;
    errMsg += sqlite3_errmsg(_db);
    fatal0(errMsg);
  }
}

bool DbManager::closeConnection()
{
  if ( _isOpen ) {

    sqlCheck(sqlite3_close(_db), "Can't close database: ");

    _isOpen = false;
    return true;
  } else {
    return false;
  }
}

unsigned int DbManager::prepareStmt(const std::string &stmt)
{
  sqlite3_stmt *stmtPtr;
  sqlCheck(sqlite3_prepare_v2(_db, stmt.c_str(), -1, &stmtPtr, 0), "Can't prepare statement: ");
  _stmtVector.push_back(stmtPtr);
  return _stmtVector.size()-1;
}

void DbManager::bindIntParameter(const unsigned int stmtNumber, const unsigned int parameterIndex, int value)
{
  sqlCheck(sqlite3_bind_int(_stmtVector[stmtNumber], parameterIndex, value), "Can't bind integer value: ");
}

int DbManager::getIntColumnValue(const unsigned int stmtNumber, const unsigned int columnIndex)
{
  return sqlite3_column_int(_stmtVector[stmtNumber], columnIndex);
}

bool DbManager::doStep(const unsigned int stmtNumber)
{
  const int err = sqlite3_step(_stmtVector[stmtNumber]);
  if ( err == SQLITE_ROW ) {
    return true;
  } else {
    return false;
  }
}
