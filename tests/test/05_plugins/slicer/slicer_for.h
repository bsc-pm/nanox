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

#ifndef _SLICER_FOR_H
#define _SLICER_FOR_H

#ifdef INVERT_LOOP_BOUNDARIES
#define EXECUTE(get_slicer,slicer_data,upper2,lower2,k_offset,step2,chunk2)\
   for ( i = 0; i < NUM_ITERS; i++ ) {\
      _loop_data.offset = -k_offset; \
      sys.loadPlugin( "slicer-" + std::string(get_slicer) ); \
      Slicer *slicer = sys.getSlicer ( get_slicer ); \
      WD * wd = new WorkDescriptor( new SMPDD( main__loop_1 ), sizeof( _loop_data ), __alignof__(nanos_loop_info_t),( void * ) &_loop_data, 0, NULL, NULL );\
      wd->setSlicer(slicer);\
      _loop_data.loop_info.lower = lower2 + k_offset; \
      _loop_data.loop_info.upper = upper2 + k_offset; \
      _loop_data.loop_info.step = step2; \
      _loop_data.loop_info.chunk = chunk2; \
      WD *wg = getMyThreadSafe()->getCurrentWD();\
      wg->addWork( *wd );\
      if ( sys.getPMInterface().getInternalDataSize() > 0 ) { \
         char *idata = NEW char[sys.getPMInterface().getInternalDataSize()]; \
         sys.getPMInterface().initInternalData( idata ); \
         wd->setInternalData( idata ); \
      } \
      sys.setupWD( *wd, (nanos::WD *) wg );\
      sys.submit( *wd );\
      wg->waitCompletion();\
      if (step2 > 0 ) for ( int j = lower2+k_offset; j <= upper2+k_offset; j+= step2 ) A[j+_loop_data.offset]--; \
      else if ( step2 < 0 ) for ( int j = lower2+k_offset; j >= upper2+k_offset; j+= step2 ) A[j+_loop_data.offset]--; \
   }
#else
#define EXECUTE(get_slicer,slicer_data,lower2,upper2,k_offset,step2,chunk2)\
   for ( i = 0; i < NUM_ITERS; i++ ) {\
      _loop_data.offset = -k_offset; \
      sys.loadPlugin( "slicer-" + std::string(get_slicer) ); \
      Slicer *slicer = sys.getSlicer ( get_slicer ); \
      WD * wd = new WorkDescriptor( new SMPDD( main__loop_1 ), sizeof( _loop_data ), __alignof__(nanos_loop_info_t),( void * ) &_loop_data, 0, NULL, NULL );\
      wd->setSlicer(slicer);\
      _loop_data.loop_info.lower = lower2 + k_offset; \
      _loop_data.loop_info.upper = upper2 + k_offset; \
      _loop_data.loop_info.step = step2; \
      _loop_data.loop_info.chunk = chunk2; \
      WD *wg = getMyThreadSafe()->getCurrentWD();\
      wg->addWork( *wd );\
      if ( sys.getPMInterface().getInternalDataSize() > 0 ) { \
         char *idata = NEW char[sys.getPMInterface().getInternalDataSize()]; \
         sys.getPMInterface().initInternalData( idata ); \
         wd->setInternalData( idata ); \
      } \
      sys.setupWD( *wd, (nanos::WD *) wg );\
      sys.submit( *wd );\
      wg->waitCompletion();\
      if (step2 > 0 ) for ( int j = lower2+k_offset; j <= upper2+k_offset; j+= step2 ) A[j+_loop_data.offset]--; \
      else if ( step2 < 0 ) for ( int j = lower2+k_offset; j >= upper2+k_offset; j+= step2 ) A[j+_loop_data.offset]--; \
   }
#endif

#define FINALIZE(type,lower,upper,offset,step,chunk)\
   print_vector();\
   if ( I[0] == STEP_ERROR ) { step_error = true; I[0] = 0;}\
   for ( i = 0; i < VECTOR_SIZE+2*VECTOR_MARGIN; i++ )\
      if ( I[i] != 0 ) {\
         if ( (i < VECTOR_MARGIN) || (i > (VECTOR_SIZE + VECTOR_MARGIN))) out_of_range = true;\
         if ( (I[i] != NUM_ITERS) && (I[i] != -NUM_ITERS)) race_condition = true;\
         I[i] = 0; check = false; p_check = false;\
      }\
   fprintf(stderr, "%s, lower (%s), upper (%s), offset (%s), step(%+03d), chunk(%02d): %s %s %s %s\n",\
      type, lower, upper, offset, step, chunk,\
      p_check?"  successful":"unsuccessful",\
      out_of_range?" - Out of Range":"",\
      race_condition?" - Race Condition":"",\
      step_error?" - Step Error":""\
   );\
   p_check = true; out_of_range = false; race_condition = false; step_error = false;


#define TEST(test_type,test_slicer_data,test_step,test_chunk)\
   EXECUTE(test_type, test_slicer_data, 0, VECTOR_SIZE, 0, +test_step, test_chunk)\
   FINALIZE (test_type,"0","+","+0000",+test_step,test_chunk)\
   EXECUTE(test_type, test_slicer_data, VECTOR_SIZE-1, -1, 0, -test_step, test_chunk)\
   FINALIZE (test_type,"+","0","+0000",-test_step,test_chunk)\
   EXECUTE(test_type, test_slicer_data, 0, VECTOR_SIZE, -VECTOR_SIZE, +test_step, test_chunk)\
   FINALIZE (test_type,"-","0","-VS  ",+test_step,test_chunk)\
   EXECUTE(test_type, test_slicer_data, VECTOR_SIZE-1, -1, -VECTOR_SIZE, -test_step, test_chunk)\
   FINALIZE (test_type,"0","-","-VS  ",-test_step,test_chunk)\
   EXECUTE(test_type, test_slicer_data, 0, VECTOR_SIZE, VECTOR_SIZE/2, +test_step, test_chunk)\
   FINALIZE (test_type,"+","+","+VS/2",+test_step,test_chunk)\
   EXECUTE(test_type, test_slicer_data, VECTOR_SIZE-1, -1, VECTOR_SIZE/2, -test_step, test_chunk)\
   FINALIZE (test_type,"+","+","+VS/2",-test_step,test_chunk)\
   EXECUTE(test_type, test_slicer_data, 0, VECTOR_SIZE, -(VECTOR_SIZE/2), +test_step, test_chunk)\
   FINALIZE (test_type,"-","+","-VS/2",+test_step,test_chunk)\
   EXECUTE(test_type, test_slicer_data, VECTOR_SIZE-1, -1, -(VECTOR_SIZE/2), -test_step, test_chunk)\
   FINALIZE (test_type,"+","-","-VS/2",-test_step,test_chunk)\
   EXECUTE(test_type, test_slicer_data, 0, VECTOR_SIZE, -(2*VECTOR_SIZE), +test_step, test_chunk)\
   FINALIZE (test_type,"-","-","-VS  ",+test_step,test_chunk)\
   EXECUTE(test_type, test_slicer_data, VECTOR_SIZE-1, -1, -(2*VECTOR_SIZE), -test_step, test_chunk)\
   FINALIZE (test_type,"-","-","-VS  ",-test_step,test_chunk)

#define TEST_SLICER(test_slicer_type, test_slicer_slicer_data)  \
   TEST(test_slicer_type, test_slicer_slicer_data, NUM_A, NUM_A)\
   TEST(test_slicer_type, test_slicer_slicer_data, NUM_B, NUM_A)\
   TEST(test_slicer_type, test_slicer_slicer_data, NUM_C, NUM_A)\
   TEST(test_slicer_type, test_slicer_slicer_data, NUM_A, NUM_B)\
   TEST(test_slicer_type, test_slicer_slicer_data, NUM_B, NUM_B)\
   TEST(test_slicer_type, test_slicer_slicer_data, NUM_C, NUM_B)\
   TEST(test_slicer_type, test_slicer_slicer_data, NUM_A, NUM_C)\
   TEST(test_slicer_type, test_slicer_slicer_data, NUM_B, NUM_C)\
   TEST(test_slicer_type, test_slicer_slicer_data, NUM_C, NUM_C)

#endif
