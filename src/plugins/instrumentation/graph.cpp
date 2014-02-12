#include "graph_utils.hpp"

#include "instrumentation.hpp"
#include "instrumentationcontext_decl.hpp"
#include "os.hpp"
#include "plugin.hpp"
#include "smpdd.hpp"
#include "system.hpp"

#include <cassert>
#include <iostream>
#include <fstream>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

namespace nanos {

class InstrumentationGraphInstrumentation: public Instrumentation
{
   private:
      // Shared structures relating tasks with it identifier, execution time and color
      std::map<int64_t, std::string> _funct_id_to_color_map;      /*!< relation between a task id and its color */
      std::map<int64_t, std::string> _funct_id_to_funct_decl_map; /*!< relation between a task id and its name */
      std::map<int, double> _wd_id_to_time_map;                   /*!< pair of task and time expend in execution */
      std::map<int, float> _wd_id_to_last_time_map;               /*!< last time the task has been (re)started */
      std::set<int64_t> _wd_id_independent;                       /*!< tasks which are unconnected */
      
      // Shared variables storing temporary portions of the graph that can be flushed or not into the graph
      // The will be included in the final dot only if they contain nodes
      std::string _tmp_taskwait;                /*!< temporary dot generated for a taskwait group of tasks */
      std::string _tmp_workdescriptor;          /*!< temporary dot generated for a workdescriptor group of tasks */
      
      // Members storing information for clustering nodes
      unsigned int _last_tw;                    /*!< identifier of the last taskwait group */
      unsigned int _last_id;
      
      // Members for inter-cluster connexion
      std::string _last_tw_cluster;             /*!< name of the last flushed taskwait cluster */
      std::string _current_tw_cluster;          /*!< name of the taskwait cluster we are included */
      unsigned int _last_task_id;               /*!< identifier of the last genrated workdescriptor */
      std::string _inter_cluster_edges;         /*!< temporary string containing the inter-cluster relations */
      
      // Dot file and formatting members
      std::string _indent;          /*!< indentation that has to be applied to the next line of dot file */ 
      std::string _file_name;       /*!< name of the dot file without the extension. It is also used to generate the pdf */
      std::ofstream _dot_file;      /*!< stream storing the dot file contents */

#ifndef NANOS_INSTRUMENTATION_ENABLED
   public:
      // constructor
      InstrumentationGraphInstrumentation() : Instrumentation(),
                                              _funct_id_to_color_map( ), _funct_id_to_funct_decl_map( ),
                                              _wd_id_to_time_map( ), _wd_id_to_last_time_map( ), _wd_id_independent( ),
                                              _tmp_taskwait(""), _tmp_workdescriptor(""), _last_tw(1), _last_id(1),
                                              _last_tw_cluster(""), _current_tw_cluster(""),
                                              _last_task_id(0), _inter_cluster_edges(""), 
                                              _indent(""), _file_name(""), _dot_file( ) {}
      // destructor
      ~InstrumentationGraphInstrumentation() {}

      // low-level instrumentation interface (mandatory functions)
      void initialize( void ) {}
      void finalize( void ) {}
      void disable( void ) {}
      void enable( void ) {}
      void addResumeTask( WorkDescriptor &w ) {}
      void addSuspendTask( WorkDescriptor &w, bool last ) {}
      void addEventList ( unsigned int count, Event *events ) {}
      void threadStart( BaseThread &thread ) {}
      void threadFinish ( BaseThread &thread ) {}
#else
   public:
      // constructor
      InstrumentationGraphInstrumentation() : Instrumentation( *new InstrumentationContextDisabled() ),
                                              _funct_id_to_color_map( ), _funct_id_to_funct_decl_map( ),
                                              _wd_id_to_time_map( ), _wd_id_to_last_time_map( ), _wd_id_independent( ),
                                              _tmp_taskwait(""), _tmp_workdescriptor(""), _last_tw(1), _last_id(1),
                                              _last_tw_cluster(""), _current_tw_cluster(""),
                                              _last_task_id(0), _inter_cluster_edges(""), 
                                              _indent(""), _file_name(""), _dot_file( ) {}
      // destructor
      ~InstrumentationGraphInstrumentation ( ) {}

      // low-level instrumentation interface (mandatory functions)
      void initialize( void )
      {
         // Generate the name of the dot file from the name of the binary
         _file_name = OS::getArg( 0 );
         size_t slash_pos = _file_name.find_last_of("/");
         if( slash_pos != std::string::npos )
            _file_name = _file_name.substr( slash_pos+1, _file_name.size( ) - slash_pos );
         std::stringstream ss; ss << getpid();
         _file_name = _file_name + "_" + ss.str();
         
         // Open the file and start the graph
         _dot_file.open( std::string( _file_name + ".dot" ).c_str( ) );
         if( _dot_file.is_open( ) )
         {  // Open the graph in the dot file
            _dot_file << "digraph {\n";
            _indent += "  ";
         }
         else
         {
            exit( EXIT_FAILURE );
         }
      }

      void finalize( void )
      {
         if( !_funct_id_to_funct_decl_map.empty( ) )
         {
            // /////////////////////////// Print the size of the nodes /////////////////////////// //
            double time_avg = 0;
            {
               std::map<int, double>::iterator it = _wd_id_to_time_map.begin( );
               for( ; it != _wd_id_to_time_map.end( ); ++it )
                  time_avg += it->second;
               time_avg /= _wd_id_to_time_map.size( );
            }
            if( time_avg == 0 )
            {
               std::map<int, double>::iterator it = _wd_id_to_time_map.begin( );
               for( ; it != _wd_id_to_time_map.end( ); ++it )
                  _dot_file << "  " << it->first << "[width=1, height=1];\n";
            }
            else
            {
               std::map<int, double>::iterator it = _wd_id_to_time_map.begin( );
               for( ; it != _wd_id_to_time_map.end( ); ++it )
               {
                  double size = std::max( 0.01, std::abs( (double)(it->second/time_avg) ) );
                  _dot_file << "  " << it->first << "[width=" << size << ", height=" << size << "];\n";
               }
            }
            
            
            // ////////////////////////// Print the inter-cluster edges ////////////////////////// //
            _dot_file << _inter_cluster_edges;
            
            
            // //////////////////////////////// Print the legend ///////////////////////////////// //
            assert( _funct_id_to_funct_decl_map.size( ) == _funct_id_to_color_map.size( ) );
            
            // Nodes legend
            _dot_file << "  subgraph cluster0 {\n";
            _dot_file << "    label=\"User functions:\"; style=\"rounded\"; rankdir=\"TB\";\n";
            int id = -1;
            std::set<std::string> printed_task_names;
            std::map<int64_t, std::string>::iterator it_name = _funct_id_to_funct_decl_map.begin( );
            for( ; it_name != _funct_id_to_funct_decl_map.end( ) ; ++it_name )
            {
               if( printed_task_names.find( it_name->second ) == printed_task_names.end( ) )
               {
                  _dot_file << "    subgraph {\n";
                  _dot_file << "      rank=same;\n";
                  _dot_file << "      " << it_name->second << "[color=\"white\", margin=\"0.0,0.0\"];\n";
                  // All tasks with the same name must be printed in the same subgraph
                  printed_task_names.insert( it_name->second );
                  int last_id = 0;
                  std::map<int64_t, std::string>::iterator it_color = _funct_id_to_color_map.begin( );
                  for( std::map<int64_t, std::string>::iterator it_name_2 = _funct_id_to_funct_decl_map.begin( ); 
                       ( it_name_2 != _funct_id_to_funct_decl_map.end( ) ) && ( it_color != _funct_id_to_color_map.end( ) ); 
                       ++it_name_2, ++it_color )
                  {
                     if( it_name_2->second == it_name->second )
                     {
                        _dot_file << "      " << id << "[label=\"\",  width=0.3, height=0.3, shape=box, "
                                  <<        "fillcolor=" << _funct_id_to_color_map[it_color->first].c_str( ) << ", style=filled];\n";
                        if( last_id != 0 )
                           _dot_file << "      " << last_id << "->" << id << "[style=\"invis\"];\n";
                        last_id = id;
                        id--;
                     }
                  }
                  _dot_file << "      " << last_id << "->" << it_name->second << "[style=\"invis\"];\n";
                  _dot_file << "    }\n";
               }
            }
            // We want the pairs of <task_set, task_name> to be shown vertically
            // To achieve it, we create edges between each two consecutive pair subgraph
            std::set<std::string>::iterator it2;
            for( std::set<std::string>::iterator it = printed_task_names.begin( ); it != printed_task_names.end( ); ++it )
            {
               it2 = it; it2++;
               if( it2 != printed_task_names.end( ) )
                  _dot_file << "    " << *it << "->" << *it2 << "[style=\"invis\"];\n";
            }
            
            std::set<std::string>::iterator it_task = printed_task_names.begin( );
            std::set<std::string>::iterator it_task2 = it_task; ++it_task2;
            for( ; it_task2 != printed_task_names.end( ); ++it_task, ++it_task2 )
               _dot_file << "    " << *it_task << "->" << *it_task2 << "[style=\"invis\"];\n";
            
            // Close the node legend subgraph
            _dot_file << "  }\n";

            // Edges legend
            _dot_file << "  subgraph cluster1 {\n";
            _dot_file << "    label=\"Dependence type:\"; style=\"rounded\"; rankdir=\"TB\";\n";
            _dot_file << "    subgraph A{\n";
            _dot_file << "      rank=same;\n";
            _dot_file << "      \"solid line\"[label=\"\", color=\"white\", shape=\"point\"];\n";
            _dot_file << "      \"True dependence\"[color=\"white\", margin=\"0.0,0.0\"];\n";
            _dot_file << "      \"solid line\"->\"True dependence\"[minlen=2.0];\n";
            _dot_file << "    }\n";
            _dot_file << "    subgraph B{\n";
            _dot_file << "      rank=same;\n";
            _dot_file << "      \"dashed line\"[label=\"\", color=\"white\", shape=\"point\"];\n";
            _dot_file << "      \"Anti-dependence\"[color=\"white\", margin=\"0.0,0.0\"];\n";
            _dot_file << "      \"dashed line\"->\"Anti-dependence\"[style=\"dashed\", minlen=2.0];\n";
            _dot_file << "    }\n";
            _dot_file << "    subgraph C{\n";
            _dot_file << "      rank=same;\n";
            _dot_file << "      \"dotted line\"[label=\"\", color=\"white\", shape=\"point\"];\n";
            _dot_file << "      \"Output dependence\"[color=\"white\", margin=\"0.0,0.0\"];\n";
            _dot_file << "      \"dotted line\"->\"Output dependence\"[style=\"dotted\", minlen=2.0];\n";
            _dot_file << "    }\n";
            _dot_file << "    \"solid line\"->\"dashed line\"[ltail=A, lhead=B, style=\"invis\"];\n";
            _dot_file << "    \"dashed line\"->\"dotted line\"[ltail=B, lhead=C, style=\"invis\"];\n";
            _dot_file << "  }\n";
         }

         
         // ///////////////////////// Close the graph and the dot file //////////////////////// //
         // 
         _dot_file << "}\n";
         _dot_file.close( );

         
         // //////////////////// Create the graph image from the dot file ///////////////////// //
         // 
         std::string command = "dot -Tpdf " + _file_name + ".dot -o " + _file_name + ".pdf";
         if ( system( command.c_str( ) ) != 0 )
            warning( "Could not create the pdf file." );
      }

      void disable( void ) {}

      void enable( void ) {}

      static double get_current_time( )
      {
         struct timeval tv;
         gettimeofday(&tv,0);
         return ( ( double ) tv.tv_sec*1000000L ) + ( ( double )tv.tv_usec );
      }

      void addResumeTask( WorkDescriptor &w )
      {
         int wd_id = w.getId();

         std::stringstream wd_s; wd_s << wd_id << "_" << _last_id++;
         _tmp_workdescriptor += _indent + "subgraph cluster_wd" + wd_s.str( ) + " {\n";
         _indent += "  ";
         if( wd_id != 1 ) {
            _tmp_workdescriptor += _indent + "label=\"" + "other" + "\" style=\"rounded\"; rankdir=\"TB\";";
            _wd_id_to_last_time_map[wd_id] = get_current_time( );
         } else {
            _tmp_workdescriptor += _indent + "label=\"" + "main" + "\" style=\"rounded\"; rankdir=\"TB\";";
         }
         
         std::stringstream ss; ss << _last_tw++;
         _last_tw_cluster = _current_tw_cluster;
         _current_tw_cluster = "cluster_tw" + ss.str( );
         _tmp_taskwait += _indent + "subgraph " + _current_tw_cluster + " {\n";
         _indent += "  ";
         _tmp_taskwait += _indent + "label=\"\";";
      }

      void addSuspendTask( WorkDescriptor &w, bool last )
      {
         int wd_id = w.getId();
         
         _indent = _indent.substr( 0, _indent.size( )-2 );
         if( !_tmp_taskwait.empty( ) )
            _tmp_taskwait.clear( );
         else
            _dot_file << _indent << "}\n";
         
         _indent = _indent.substr( 0, _indent.size( )-2 );
         if( !_tmp_workdescriptor.empty( ) )
            _tmp_workdescriptor.clear( );
         else
            _dot_file << _indent << "}\n";
         
         if( wd_id != 1 )
         {
            double last_time = _wd_id_to_last_time_map[wd_id];
            double current_time = get_current_time( );
            double time = last_time - current_time;
            if( _wd_id_to_time_map.find( wd_id ) == _wd_id_to_time_map.end( ) )
               _wd_id_to_time_map[wd_id] = time;
            else
               _wd_id_to_time_map[wd_id] += time;
         }
      }

      void addEventList ( unsigned int count, Event *events )
      {
         InstrumentationDictionary *iD = getInstrumentationDictionary();
         static const nanos_event_key_t create_wd_id = iD->getEventKey("create-wd-id");
         static const nanos_event_key_t create_wd_ptr = iD->getEventKey("create-wd-ptr");
         static const nanos_event_key_t dependence = iD->getEventKey("dependence");
         static const nanos_event_key_t dep_direction = iD->getEventKey("dep-direction");
         static const nanos_event_key_t user_funct_location = iD->getEventKey("user-funct-location");
         static const nanos_event_key_t taskwait = iD->getEventKey("taskwait");

         unsigned int i;
         for( i=0; i<count; i++) {
            Event &e = events[i];
            if ( e.getKey( ) == dependence )
            {  // A dependence occurs
               unsigned sender = (e.getValue( ) >> 32) & 0xFFFFFFFF;
               unsigned receiver = e.getValue( ) & 0xFFFFFFFF;

               e = events[++i];
               assert( e.getKey( ) == dep_direction );
               if( e.getValue() == 1 )
               {  // True dependence
                  _dot_file << _indent << sender << " -> " << receiver << ";\n";
               }
               else if( e.getValue( ) == 2 )
               {  // Anti-dependence
                  _dot_file << _indent << sender << " -> " << receiver << " [style=dashed];\n";
               }
               else if( e.getValue( ) == 3 )
               {  // Output dependence
                  _dot_file << _indent << sender << " -> " << receiver << " [style=dotted];\n";
               }
               else if( e.getValue( ) == 4 )
               {  // Output dependence
                  _dot_file << _indent << sender << " -> d" << receiver << ";\n";
               }
               else if( e.getValue( ) == 5 )
               {  // Output dependence
                  _dot_file << _indent << "  d" << sender << " -> " << receiver << ";\n";
               }
               
               _wd_id_independent.erase( sender );
               _wd_id_independent.erase( receiver );
            }
            else if( e.getKey( ) == create_wd_ptr )
            {  // A wd is submitted
               WorkDescriptor *wd = (WorkDescriptor *) e.getValue();
               int64_t funct_id = (int64_t) ((ext::SMPDD &)(wd->getActiveDevice())).getWorkFct();

               std::string color = "";
               if( _funct_id_to_color_map.find( funct_id ) == _funct_id_to_color_map.end( ) )
               {
                  color = wd_to_color_hash( funct_id );
                  _funct_id_to_color_map[funct_id] = color ;
               }
               else
               {
                  color = _funct_id_to_color_map[funct_id];
               }

               e = events[--i];
               assert(e.getKey() == create_wd_id);
               int64_t wd_id = e.getValue();
               
               if( !_tmp_workdescriptor.empty( ) )
               {
                  _dot_file << _tmp_workdescriptor << "\n";
                  _tmp_workdescriptor.clear( );
               }
               if( !_tmp_taskwait.empty( ) )
               {
                  _dot_file << _tmp_taskwait << "\n";
                  _tmp_taskwait.clear( );
               }
               _dot_file << _indent << wd_id << "[fillcolor=" << color.c_str( ) << ", style=filled];\n";
               if( !_last_tw_cluster.empty( ) )
               {
                  std::stringstream t_s; t_s << _last_task_id;
                  std::stringstream wd_s; wd_s << wd_id;
                  std::string temp = "  " + t_s.str( ) + " -> " + wd_s.str( )
                                   + "[ltail=" + _current_tw_cluster + ", lhead=" + _last_tw_cluster + "];\n";
                  _inter_cluster_edges += temp;
                  _last_tw_cluster.clear( );
               }
               _last_task_id = wd_id;
               
               _wd_id_independent.insert( wd_id );
            }
            else if ( e.getKey( ) == user_funct_location )
            {
               if( e.getValue( ) != 0
                   && _funct_id_to_funct_decl_map.find( e.getValue( ) ) == _funct_id_to_funct_decl_map.end( ) )
               {
                  std::string description = iD->getValueDescription( user_funct_location, e.getValue( ) );
                  int pos2 = description.find_first_of( "(" );
                  int pos1 = description.find_last_of ( " ", pos2 );
                  _funct_id_to_funct_decl_map[ e.getValue( ) ] = '\"' + description.substr( pos1+1, pos2-pos1-1 ) + '\"';
               }
            }
            else if ( e.getKey( ) == taskwait )
            {
               _indent = _indent.substr( 0, _indent.size( )-2 );
               
               if( !_tmp_taskwait.empty( ) ) {
                  _tmp_taskwait.clear( );
               } else {
                  _dot_file << _indent << "}\n";
               }
               
               std::stringstream ss; ss << _last_tw++;
               _last_tw_cluster = _current_tw_cluster;
               _current_tw_cluster = "cluster_tw_" + ss.str( );
               _tmp_taskwait += _indent + "subgraph " + _current_tw_cluster + " {\n";
               _indent += "  ";
               _tmp_taskwait += _indent + "label=\"\";\n";
            }
         }
      }

      void threadStart( BaseThread &thread ) {}

      void threadFinish ( BaseThread &thread ) {}

#endif

};

namespace ext {

class InstrumentationGraphInstrumentationPlugin : public Plugin {
   public:
      InstrumentationGraphInstrumentationPlugin () : Plugin("Instrumentation which print the trace to std out.",1) {}
      ~InstrumentationGraphInstrumentationPlugin () {}

      void config( Config &cfg ) {}

      void init ()
      {
         sys.setInstrumentation( new InstrumentationGraphInstrumentation() );
      }
};

} // namespace ext

} // namespace nanos

DECLARE_PLUGIN("intrumentation-print_trace",nanos::ext::InstrumentationGraphInstrumentationPlugin);
