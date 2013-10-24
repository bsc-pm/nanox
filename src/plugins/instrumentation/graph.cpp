#include "graph_utils.hpp"

#include "plugin.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "instrumentationcontext_decl.hpp"
#include "smpdd.hpp"

#include <iostream>
#include <fstream>
#include <stdlib.h>

#include <cassert>
#include <map>
#include <math.h>

#include <sys/time.h>
#include <time.h>

#define dot_file_name "graph.dot"
#define pdf_file_name "graph.pdf"

namespace nanos {

class InstrumentationGraphInstrumentation: public Instrumentation
{
   private:
      std::map<int64_t, std::string> _funct_id_to_color_map;
      std::map<int64_t, std::string> _funct_id_to_funct_decl_map;
      std::map<int, double> _wd_id_to_time_map;
      std::map<int, float> _wd_id_to_last_time_map;
      std::set<unsigned> _wd_id_independent;
      std::ofstream _dot_file;
      unsigned int _last_tw;
      std::string _indent;

#ifndef NANOS_INSTRUMENTATION_ENABLED
   public:
      // constructor
      InstrumentationGraphInstrumentation() : Instrumentation(),
                                              _funct_id_to_color_map( ), _funct_id_to_funct_decl_map( ),
                                              _wd_id_to_time_map( ), _wd_id_to_last_time_map( ),
                                              _wd_id_independent( ), _dot_file( dot_file_name ), 
                                              _last_tw(1), _indent("") {}
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
                                              _wd_id_to_time_map( ), _wd_id_to_last_time_map( ),
                                              _wd_id_independent( ), _dot_file( dot_file_name ), 
                                              _last_tw(1), _indent("") {}
      // destructor
      ~InstrumentationGraphInstrumentation ( ) {}

      // low-level instrumentation interface (mandatory functions)
      void initialize( void )
      {
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
         _dot_file << _indent << "}\n";
         _indent = _indent.substr( 0, _indent.size( )-2 );
         _dot_file << _indent << "}\n";
         _indent = _indent.substr( 0, _indent.size( )-2 );
         if( !_funct_id_to_funct_decl_map.empty( ) )
         {
            // Print the size of the nodes
            double time_avg = 0;
            for( std::map<int, double>::iterator it = _wd_id_to_time_map.begin( );
                 it != _wd_id_to_time_map.end( ); ++it )
            {
                time_avg += it->second;
            }
            time_avg /= _wd_id_to_time_map.size( );
            if( time_avg == 0 )
            {
                for( std::map<int, double>::iterator it = _wd_id_to_time_map.begin( );
                    it != _wd_id_to_time_map.end( ); ++it )
                {
                    _dot_file << "  " << it->first << "[width=1, height=1];\n";
                }
            }
            else
            {
                for( std::map<int, double>::iterator it = _wd_id_to_time_map.begin( );
                    it != _wd_id_to_time_map.end( ); ++it )
                {
                    double size = std::max( 0.01, std::abs( (double)(it->second/time_avg) ) );
                    _dot_file << "  " << it->first << "[width=" << size << ", height=" << size << "];\n";
                }
            }

            // Align nodes depending on the moment they have been executed
            for( std::set<unsigned>::iterator it = _wd_id_independent.begin( ); it != _wd_id_independent.end( ); ++it )
            {
                _dot_file << "  {rank=same; " << *it << " " << ( *it - 1 ) << "}\n";
            }

            // Print the legend
            assert( _funct_id_to_funct_decl_map.size( ) == _funct_id_to_color_map.size( ) );
                    // Nodes legend
            _dot_file << "  subgraph cluster0 {\n";
            _dot_file << "    label=\"User functions:\"; style=\"rounded\"; rankdir=\"TB\";\n";
            std::map<int64_t, std::string>::iterator it_name = _funct_id_to_funct_decl_map.begin( );
            int id = -1;
            std::set<std::string> printed_task_names;
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
            std::set<std::string>::iterator it2;
            for( std::set<std::string>::iterator it = printed_task_names.begin( ); it != printed_task_names.end( ); ++it )
            {
                it2 = it; it2++;
                if( it2 != printed_task_names.end( ) )
                    _dot_file << "    " << *it << "->" << *it2 << "[style=\"invis\"];\n";
            }

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
            _dot_file << "    \"solid line\"->\"dashed line\"[ltail=A, lhead=B, style=\"invis\"];";
            _dot_file << "    \"dashed line\"->\"dotted line\"[ltail=B, lhead=C, style=\"invis\"];";
            _dot_file << "  }\n";
         }

         // Close the graph in the dot file
         _dot_file << "}\n";
         _dot_file.close( );

         // Create the graph image from the dot file
         std::string command = "dot -Tpdf " + std::string( dot_file_name ) + " -o " + std::string( pdf_file_name );
         if ( system( command.c_str( ) ) != 0 ) {
            warning( "Could not create the pdf file" );
         }
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

         _dot_file << _indent << "subgraph cluster_wd_" << wd_id << " {\n";
          _indent += "  ";
          if( wd_id != 1 )
          {
            _dot_file << _indent << "label=\"" << "other" << "\" style=\"rounded\"; rankdir=\"TB\";\n";
             _wd_id_to_last_time_map[wd_id] = get_current_time( );
          } else {
            _dot_file << _indent << "label=\"" << "main" << "\" style=\"rounded\"; rankdir=\"TB\";\n";
          }
         _dot_file << _indent << "subgraph cluster_tw_" << _last_tw++ << " {\n";
         _indent += "  ";
         _dot_file << _indent << "label=\"\"\n";
      }

      void addSuspendTask( WorkDescriptor &w, bool last )
      {

         int wd_id = w.getId();

         _indent = _indent.substr( 0, _indent.size( )-2 );
         _dot_file << _indent << "}\n";
         _indent = _indent.substr( 0, _indent.size( )-2 );
         _dot_file << _indent << "}\n";
         
         if( wd_id != 1 )
         {
            double last_time = _wd_id_to_last_time_map[wd_id];
            double current_time = get_current_time( );
            double time = last_time - current_time;
            if( _wd_id_to_time_map.find( wd_id ) == _wd_id_to_time_map.end( ) )
            {
                _wd_id_to_time_map[wd_id] = time;
            }
            else
            {
                _wd_id_to_time_map[wd_id] += time;
            }
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
               {    // True dependence
                   _dot_file << _indent << sender << " -> " << receiver << ";\n";
               }
               else if( e.getValue( ) == 2 )
               {    // Anti-dependence
                   _dot_file << _indent << sender << " -> " << receiver << " [style=dashed];\n";
               }
               else if( e.getValue( ) == 3 )
               {    // Output dependence
                   _dot_file << _indent << sender << " -> " << receiver << " [style=dotted];\n";
               }
               else if( e.getValue( ) == 4 )
               {    // Output dependence
                   _dot_file << _indent << sender << " -> d" << receiver << ";\n";
               }
               else if( e.getValue( ) == 5 )
               {    // Output dependence
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
               _dot_file << _indent << wd_id << "[fillcolor=" << color.c_str( ) << ", style=filled];\n";
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
               _dot_file << _indent << "}\n";
               _dot_file << _indent << "subgraph cluster_tw_" << _last_tw++ << " {\n";
               _indent += "  ";
               _dot_file << _indent << "label=\"\"\n";
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
