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

#include "dbmanager_sqlite3.hpp"
#include "debug.hpp"

using namespace nanos;

SQLite3DbManager::~SQLite3DbManager()
{
   fatal0("SQLite3DbManager: empty class compiled")
}

bool SQLite3DbManager::openConnection(const std::string &databaseName)
{
   fatal0("SQLite3DbManager: empty class compiled")
   return false;
}

void SQLite3DbManager::cleanPreparedStmts()
{
   fatal0("SQLite3DbManager: empty class compiled")
}

bool SQLite3DbManager::openConnection()
{
   fatal0("SQLite3DbManager: empty class compiled")
   return false;
}

void SQLite3DbManager::sqlCheck(const int err, const std::string &msg)
{
   fatal0("SQLite3DbManager: empty class compiled")
}

bool SQLite3DbManager::closeConnection()
{
   fatal0("SQLite3DbManager: empty class compiled")
   return false;
}

unsigned int SQLite3DbManager::prepareStmt(const std::string &stmt)
{
   fatal0("SQLite3DbManager: empty class compiled")
   return 1;
}

void SQLite3DbManager::bindIntParameter(const unsigned int stmtNumber, const unsigned int parameterIndex, int value)
{
   fatal0("SQLite3DbManager: empty class compiled")
}

void SQLite3DbManager::bindInt64Parameter(const unsigned int stmtNumber, const unsigned int parameterIndex, long long int value)
{
   fatal0("SQLite3DbManager: empty class compiled")
}

int SQLite3DbManager::getIntColumnValue(const unsigned int stmtNumber, const unsigned int columnIndex)
{
   fatal0("SQLite3DbManager: empty class compiled")
}

bool SQLite3DbManager::doStep(const unsigned int stmtNumber)
{
   fatal0("SQLite3DbManager: empty class compiled")
   return false;
}
