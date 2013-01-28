#include "graph_utils.hpp"

#include "plugin.hpp"
#include "system.hpp"
#include "instrumentation.hpp"
#include "instrumentationcontext_decl.hpp"
#include "smpdd.hpp"

#include <cassert>
#include <iostream>
#include <fstream>
#include <map>
#include <stdlib.h>

#define dot_file_name "graph.dot"
#define pdf_file_name "graph.pdf"

namespace nanos {

class InstrumentationGraphInstrumentation: public Instrumentation
{
   private:
      std::map<int64_t, std::string> _funct_id_to_color_map;
      std::map<int64_t, std::string> _funct_id_to_funct_decl_map;
      std::ofstream _dot_file;

#ifndef NANOS_INSTRUMENTATION_ENABLED
   public:
      // constructor
      InstrumentationGraphInstrumentation() : Instrumentation(),
                                              _funct_id_to_color_map( ), _funct_id_to_funct_decl_map( ),
                                              _dot_file( dot_file_name ) {}
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
      InstrumentationGraphInstrumentation() : Instrumentation( *new InstrumentationContext() ),
                                              _funct_id_to_color_map( ), _funct_id_to_funct_decl_map( ),
                                              _dot_file( dot_file_name ) {}
      // destructor
      ~InstrumentationGraphInstrumentation ( ) {}

      // low-level instrumentation interface (mandatory functions)
      void initialize( void )
      {
         if( _dot_file.is_open( ) )
         {  // Open the graph in the dot file
            _dot_file << "digraph {\n";
         }
         else
         {
            exit( EXIT_FAILURE );
         }
      }
      void finalize( void )
      {
         // Print the legend
         if( !_funct_id_to_funct_decl_map.empty( ) )
         {
            assert( _funct_id_to_funct_decl_map.size( ) == _funct_id_to_color_map.size( ) );

            // Nodes legend
            _dot_file << "  subgraph cluster0 {\n";
            _dot_file << "    label=\"User functions:\"; style=\"rounded\"; rankdir=\"TB\";\n";
            std::map<int64_t, std::string>::iterator it_name = _funct_id_to_funct_decl_map.begin( );
            std::map<int64_t, std::string>::iterator it_color = _funct_id_to_color_map.begin( );
            int id = -1;
            for( ; ( it_name != _funct_id_to_funct_decl_map.end( ) )
                   && ( it_color != _funct_id_to_color_map.end( ) ) ; ++it_name, ++it_color )
            {
               _dot_file << "    subgraph {\n";
               _dot_file << "      rank=same;\n";
               _dot_file << "      " << id << "[label=\"\",  width=0.3, height=0.3, shape=box, "
                         <<        "fillcolor=" << _funct_id_to_color_map[it_color->first].c_str( ) << ", style=filled];\n";
               _dot_file << "      " << it_name->second << "[color=\"white\", margin=\"0.0,0.0\"];\n";
               _dot_file << "      " << id << "->" << it_name->second << "[style=\"invis\"];\n";
               _dot_file << "    }\n";
               id--;
            }
            for( int idd = -1; idd > id+1; --idd)
            {
                _dot_file << "    " << idd << "->" << idd-1 << "[style=\"invis\"];\n";
            }

            _dot_file << "  }\n";

            // Edges legend
            _dot_file << "  subgraph cluster1 {\n";
            _dot_file << "    label=\"Dependence type:\"; style=\"rounded\"; rankdir=\"TB\";\n";
            _dot_file << "    subgraph A{\n";
            _dot_file << "      rank=same;\n";
            _dot_file << "      \"solid line\"[label=\"\", color=\"white\", shape=\"point\"];\n";
            _dot_file << "      \"Input dependence\"[color=\"white\", margin=\"0.0,0.0\"];\n";
            _dot_file << "      \"solid line\"->\"Input dependence\"[minlen=2.0];\n";
            _dot_file << "    }\n";
            _dot_file << "    subgraph B{\n";
            _dot_file << "      rank=same;\n";
            _dot_file << "      \"dashed line\"[label=\"\", color=\"white\", shape=\"point\"];\n";
            _dot_file << "      \"Output dependence\"[color=\"white\", margin=\"0.0,0.0\"];\n";
            _dot_file << "      \"dashed line\"->\"Output dependence\"[style=\"dashed\", minlen=2.0];\n";
            _dot_file << "    }\n";
            _dot_file << "    \"solid line\"->\"dashed line\"[style=\"invis\"];\n";
            _dot_file << "  }\n";
         }

         // Close the graph in the dot file
         _dot_file << "}\n";
         _dot_file.close( );

         // Create the graph image from the dot file
         std::string command = "dot -Tpdf " + std::string( dot_file_name ) + " -o " + std::string( pdf_file_name );
         system( command.c_str( )/*"dot -Tpdf graph.dot -o graph.pdf"*/ );
      }
      void disable( void ) {}
      void enable( void ) {}
      void addResumeTask( WorkDescriptor &w ) {}
      void addSuspendTask( WorkDescriptor &w, bool last ) {}
      void addEventList ( unsigned int count, Event *events )
      {
         InstrumentationDictionary *iD = getInstrumentationDictionary();
         static const nanos_event_key_t create_wd_id = iD->getEventKey("create-wd-id");
         static const nanos_event_key_t create_wd_ptr = iD->getEventKey("create-wd-ptr");
         static const nanos_event_key_t dependence = iD->getEventKey("dependence");
         static const nanos_event_key_t dep_direction = iD->getEventKey("dep-direction");
         static const nanos_event_key_t user_funct_location = iD->getEventKey("user-funct-location");

         unsigned int i;
         for( i=0; i<count; i++) {
            Event &e = events[i];
            if ( e.getKey() == dependence )
            {  // A dependence occurs
               unsigned sender = (e.getValue() >> 32) & 0xFFFFFFFF;
               unsigned receiver = e.getValue() & 0xFFFFFFFF;

               e = events[++i];
               assert( e.getKey() == dep_direction );
               if( e.getValue() == 0 )
               {    // Input dependence
                   _dot_file << "  " << sender << " -> " << receiver << ";\n";
               }
               else if( e.getValue() == 1 )
               {    // Output dependence
                   _dot_file << "  " << sender << " -> " << receiver << " [style=dashed];\n";
               }
            }
            else if( e.getKey() == create_wd_ptr )
            {  // A wd is submitted
               WorkDescriptor *wd = (WorkDescriptor *) e.getValue();
               int64_t funct_id = (int64_t) ((ext::SMPDD &)(wd->getActiveDevice())).getWorkFct();

               std::string color = "";
               if( _funct_id_to_color_map.find( funct_id ) == _funct_id_to_color_map.end( ) )
               {
                   color = wd_to_color_hash( funct_id );
                   _funct_id_to_color_map.insert( std::pair<int64_t, std::string>( funct_id, color ) );
               }
               else
               {
                   color = _funct_id_to_color_map[funct_id];
               }

               e = events[--i];
               assert(e.getKey() == create_wd_id);
               int64_t wd_id = e.getValue();
               _dot_file << "  " << wd_id << "[fillcolor=" << color.c_str( ) << ", style=filled];\n";
            }
            else if ( e.getKey() == user_funct_location )
            {
                if( _funct_id_to_funct_decl_map.find( e.getValue() ) == _funct_id_to_funct_decl_map.end( ) )
                {
                    std::string description = iD->getValueDescription( user_funct_location, e.getValue( ) );
                    int pos2 = description.find_first_of( "(" );
                    int pos1 = description.find_last_of ( " ", pos2 );
                    _funct_id_to_funct_decl_map[ e.getValue( ) ] = description.substr( pos1+1, pos2-pos1-1 );
                }
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
