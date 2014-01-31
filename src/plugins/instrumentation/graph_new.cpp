#include "graph_utils_new.hpp"

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

int taskwait_id = -1;
    
class InstrumentationNewGraphInstrumentation: public Instrumentation
{
    private:
    std::set<Node*> _graph_nodes;                           /*!< relation between a wd id and its node in the graph */
    std::map<int64_t, std::string> _funct_id_to_decl_map;   /*!< relation between a task id and its name */
    double _time_avg;
    
    inline int64_t getMyWDId( ) {
        BaseThread *current_thread = getMyThreadSafe( );
        if( current_thread == NULL ) return 0;
        else if( current_thread->getCurrentWD( ) == NULL ) return 0;
        return current_thread->getCurrentWD( )->getId( );
    }
    
    inline std::string print_node( Node* n, std::string indentation ) {
        // Get the style of the node
        std::string node_attrs = "";
        if( n->is_taskwait( ) ) {
            node_attrs += "label=\"Taskwait\", ";
        } else if( n->is_barrier( ) ) {
            node_attrs += "label=\"Barrier\", ";
        }
        node_attrs += "style=\"";
        node_attrs += ( ( n->is_taskwait( ) || n->is_barrier( ) ) ? "bold" 
                                                                  : "filled" );
        // Get the color of the node
        if( n->is_task( ) ) {
            node_attrs += "\", color=\"" + wd_to_color_hash( n->get_funct_id( ) );
        }
        
        // Get the size of the node
        if( _time_avg == 0.0 ) {
            node_attrs += "\", width=\"1\", height=\"1\"";
        }
        else {
            double size = std::max( 0.01, std::abs( (double)( n->get_total_time( ) / _time_avg ) ) );
            std::stringstream ss; ss << size;
            node_attrs += "\", width=\"" + ss.str( ) + "\", height=\"" + ss.str( ) + "\"";
        }
        
        // Build and return the whole node info
        std::stringstream ss; ss << n->get_wd_id( );
        return std::string( indentation + ss.str( ) + "[" + node_attrs + "];\n" );
    }
    
    inline std::string print_edge( Edge* e, std::string indentation ) {
        std::string edge_attrs = "style=\"";
        // Compute the style of the edge
        edge_attrs += ( ( !e->is_dependency( ) || 
                        e->is_true_dependency( ) ) ? "solid" 
                                                   : ( e->is_anti_dependency( ) ? "dashed" 
                                                                                : "dotted" ) );
        // Compute the color of the edge
        edge_attrs += "\", color=\"";
        edge_attrs += ( e->is_nesting( ) ? "gray47" 
                                         : "black" );
        edge_attrs += "\"";
        // Print the edge
        std::stringstream sss; sss << e->get_source( )->get_wd_id( );
        std::stringstream sst; sst << e->get_target( )->get_wd_id( );
        return std::string( indentation + sss.str( ) + " -> " + sst.str( ) + "[" + edge_attrs + "];\n" );
    }
    
    inline std::string print_nested_nodes( Node* n, std::string indentation ) {
        std::string nested_nodes_info = "";
        // Find all nodes which parent is 'n' and the edges connecting them is a 'Nesting' edge
        for( std::set<Node*>::iterator it = _graph_nodes.begin( ); it != _graph_nodes.end( ); ++it ) 
        {
            std::set<Edge*> entries = (*it)->get_entries( );
            for( std::set<Edge*>::iterator it2 = entries.begin( ); it2 != entries.end( ); ++it2 )
            {
                if( ( (*it2)->get_source( ) == n ) && (*it2)->is_nesting( ) ) 
                {   // This is a nested relation!
                    nested_nodes_info += print_node( *it, indentation )
                                       + print_edge( *it2, indentation );
                    // Call recursively for nodes nested to the current node
                    print_nested_nodes( *it, std::string( indentation + "  " ) );
                }
            }
        }
        return nested_nodes_info;
    }
    
    inline std::string print_edges_legend( ) {
        std::string edges_legend = "";
        edges_legend += "  subgraph cluster1 {\n";
        edges_legend += "    label=\"Edge types:\"; style=\"rounded\"; rankdir=\"TB\";\n";
        edges_legend += "    subgraph A{\n";
        edges_legend += "      rank=same;\n";
        edges_legend += "      \"solid gray line\"[label=\"\", color=\"white\", shape=\"point\"];\n";
        edges_legend += "      \"Nested task\"[color=\"white\", margin=\"0.0,0.0\"];\n";
        edges_legend += "      \"solid gray line\"->\"Nested task\"[minlen=2.0, color=gray47];\n";
        edges_legend += "    }\n";
        edges_legend += "    subgraph B{\n";
        edges_legend += "      rank=same;\n";
        edges_legend += "      \"solid black line\"[label=\"\", color=\"white\", shape=\"point\"];\n";
        edges_legend += "      \"True dependence \\n Taskwait | Barrier\"[color=\"white\", margin=\"0.0,0.0\"];\n";
        edges_legend += "      \"solid black line\"->\"True dependence \\n Taskwait | Barrier\"[minlen=2.0];\n";
        edges_legend += "    }\n";
        edges_legend += "    subgraph C{\n";
        edges_legend += "      rank=same;\n";
        edges_legend += "      \"dashed line\"[label=\"\", color=\"white\", shape=\"point\"];\n";
        edges_legend += "      \"Anti-dependence\"[color=\"white\", margin=\"0.0,0.0\"];\n";
        edges_legend += "      \"dashed line\"->\"Anti-dependence\"[style=\"dashed\", minlen=2.0];\n";
        edges_legend += "    }\n";
        edges_legend += "    subgraph D{\n";
        edges_legend += "      rank=same;\n";
        edges_legend += "      \"dotted line\"[label=\"\", color=\"white\", shape=\"point\"];\n";
        edges_legend += "      \"Output dependence\"[color=\"white\", margin=\"0.0,0.0\"];\n";
        edges_legend += "      \"dotted line\"->\"Output dependence\"[style=\"dotted\", minlen=2.0];\n";
        edges_legend += "    }\n";
        edges_legend += "    \"solid gray line\"->\"solid black line\"[ltail=A, lhead=B, style=\"invis\"];\n";
        edges_legend += "    \"solid black line\"->\"dashed line\"[ltail=B, lhead=C, style=\"invis\"];\n";
        edges_legend += "    \"dashed line\"->\"dotted line\"[ltail=C, lhead=D, style=\"invis\"];\n";
        edges_legend += "  }\n";
        return edges_legend;
    }
    
    inline std::string print_nodes_legend( ) {
        std::string nodes_legend = "";
        nodes_legend += "  subgraph cluster0 {\n";
        nodes_legend += "    label=\"User functions:\"; style=\"rounded\"; rankdir=\"TB\";\n";
        
        int id = 1;
        std::set<std::string> printed_funcs;
        for( std::map<int64_t, std::string>::iterator it = _funct_id_to_decl_map.begin( ); it != _funct_id_to_decl_map.end( ) ; ++it )
        {
            if( printed_funcs.find( it->second ) == printed_funcs.end( ) )
            {
                printed_funcs.insert( it->second );
                
                nodes_legend += "    subgraph {\n";
                nodes_legend += "      rank=same;\n";
                // Print the transparent node with the name of the function
                nodes_legend += "      " + _funct_id_to_decl_map[it->first] + "[color=\"white\", margin=\"0.0,0.0\"];\n";
                // Print one node for each function id that has the same name as the current function name
                int last_id = 0;
                for( std::map<int64_t, std::string>::iterator it2 = _funct_id_to_decl_map.begin( ); 
                     it2 != _funct_id_to_decl_map.end( ); ++it2 )
                {
                    if( it2->second == it->second )
                    {
                        std::stringstream ss; ss << id;
                        nodes_legend += "      0" + ss.str( ) + "[label=\"\",  width=0.3, height=0.3, shape=box, "
                                      + "fillcolor=" + wd_to_color_hash( it->first ) + ", style=filled];\n";
                        if( last_id != 0 ) {
                            std::stringstream ss2; ss2 << last_id;
                            nodes_legend += "      0" + ss2.str( ) + " -> 0" + ss.str( ) + "[style=\"invis\"];\n";
                        }
                        last_id = id;
                        ++id;
                    }
                }
                // Print the edge between the last function id node and the name of the function
                std::stringstream ss; ss << last_id;
                nodes_legend += "      0" + ss.str( ) + "->" + it->second + "[style=\"invis\"];\n";
                nodes_legend += "    }\n";
            }
        }
    
        // We want the pairs of <task_set, task_name> to be shown vertically
        // To achieve it, we create edges between each two consecutive pair subgraph
        std::set<std::string>::iterator it2;
        for( std::set<std::string>::iterator it = printed_funcs.begin( ); it != printed_funcs.end( ); ++it )
        {
            it2 = it; it2++;
            if( it2 != printed_funcs.end( ) )
                nodes_legend += "    " + *it + " -> " + *it2 + "[style=\"invis\"];\n";
        }
        
        // Close the node legend subgraph
        nodes_legend += "  }\n";
        return nodes_legend;
    }
    
    inline Node* find_node_from_wd_id( int64_t wd_id ) {
        Node* result = NULL;
        for( std::set<Node*>::iterator it = _graph_nodes.begin( ); it != _graph_nodes.end( ); ++it ) {
            if( (*it)->get_wd_id( ) == wd_id ) {
                result = *it;
                break;
            }
        }
        return result;
    }
    
#ifndef NANOS_INSTRUMENTATION_ENABLED
    public:
    // constructor
    InstrumentationNewGraphInstrumentation() : Instrumentation(),
                                               _graph_nodes( ), _funct_id_to_decl_map( ), 
                                               _time_avg( 0.0 )
    {}
    
    // destructor
    ~InstrumentationNewGraphInstrumentation() {}

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
    InstrumentationNewGraphInstrumentation() : Instrumentation( *new InstrumentationContextDisabled() ),
                                               _graph_nodes( ), _funct_id_to_decl_map( ), 
                                               _time_avg( 0.0 )
    {}
    
    // destructor
    ~InstrumentationNewGraphInstrumentation ( ) {}

    // low-level instrumentation interface (mandatory functions)
    void initialize( void ) {}
    
    void finalize( void )
    {
        // Generate the name of the dot file from the name of the binary
        std::string file_name = OS::getArg( 0 );
        size_t slash_pos = file_name.find_last_of( "/" );
        if( slash_pos != std::string::npos )
            file_name = file_name.substr( slash_pos+1, file_name.size( )-slash_pos );
        std::stringstream ss; ss << getpid( );
        file_name = file_name + "_" + ss.str( ) + ".dot";
        
        // Open the file and start the graph
        std::ofstream dot_file;
        dot_file.open( file_name.c_str( ) );
        if( !dot_file.is_open( ) )
            exit( EXIT_FAILURE );
        
        // Compute the time average to print the nodes size accordingly
        for( std::set<Node*>::iterator it = _graph_nodes.begin( ); it != _graph_nodes.end( ); ++it ) {
            _time_avg += (*it)->get_total_time( );
        }
        _time_avg /= _graph_nodes.size( );
        
        // Print the graph
        dot_file << "digraph {\n";
        // Print the graph nodes
        for( std::set<Node*>::iterator it = _graph_nodes.begin( ); it != _graph_nodes.end( ); ++it )
        {
            if( (*it)->is_printed( ) )
                continue;
            
            // Print the current node
            std::string node_info = print_node( (*it), /*indentation*/"    " );
            (*it)->set_printed( );
            // Print all nested nodes
            std::string nested_nodes_info = print_nested_nodes( (*it), /*indentation*/"    " );
            
            if( nested_nodes_info.empty( ) ) {
                dot_file << node_info;
            } else {
                // We want all nodes nested in a task to be printed horizontally
                dot_file << "  subgraph {\n";
                dot_file << "    rank=same;\n";
                dot_file << node_info;
                dot_file << nested_nodes_info;
                dot_file << "  }\n";
            }
            
            // Print the exit edges ( outside the rank, so they are displayed top-bottom )
            std::set<Edge*> exits = (*it)->get_exits( );
            for( std::set<Edge*>::iterator edge = exits.begin( ); edge != exits.end( ); ++edge ) {
                if( !(*edge)->is_nesting( ) )   // nesting edges have been printed previously in 'print_nested_nodes'
                    dot_file << print_edge( *edge, /*indentation*/"  " );
            }
        }
        // Print the legends
        dot_file << print_nodes_legend( );
        dot_file << print_edges_legend( );
        dot_file << "}";
        
        printf( "Task Dependency Graph printed to file %s\n", file_name.c_str( ) );
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
        Node* n = find_node_from_wd_id( w.getId( ) );
        if( n != NULL ) {
            n->set_last_time( get_current_time( ) );
        }
    }

    void addSuspendTask( WorkDescriptor &w, bool last )
    {
        Node* n = find_node_from_wd_id( w.getId( ) );
        if( n != NULL ) {
            double time = n->get_last_time( ) - get_current_time( );
            n->add_total_time( time );
        }
    }

    void addEventList( unsigned int count, Event *events )
    {
        InstrumentationDictionary *iD = getInstrumentationDictionary( );
        static const nanos_event_key_t create_wd_id = iD->getEventKey( "create-wd-id" );
        static const nanos_event_key_t create_wd_ptr = iD->getEventKey( "create-wd-ptr" );
        static const nanos_event_key_t dependence = iD->getEventKey( "dependence" );
        static const nanos_event_key_t dep_direction = iD->getEventKey( "dep-direction" );
        static const nanos_event_key_t user_funct_location = iD->getEventKey( "user-funct-location" );
        static const nanos_event_key_t taskwait = iD->getEventKey( "taskwait" );
        
        // Get the node corresponding to the wd_id calling this function
        // This node won't exist if the calling wd corresponds to that of the master thread
        int64_t current_wd_id = getMyWDId( );
        Node* current_node = find_node_from_wd_id( current_wd_id );
        
        unsigned int i;
        for( i=0; i<count; i++ ) {
            Event &e = events[i];
            if( e.getKey( ) == create_wd_ptr )
            {  // A wd is submitted => create a new node
                
                // Get the identifier of the task function
                WorkDescriptor *wd = (WorkDescriptor *) e.getValue();
                int64_t funct_id = (int64_t) ((ext::SMPDD &)(wd->getActiveDevice())).getWorkFct();
                
                // Get the identifier of the wd
                e = events[--i];
                assert( e.getKey( ) == create_wd_id );
                int64_t wd_id = e.getValue( );
                
                // Create the new node
                Node* new_node = new Node( wd_id, funct_id, TaskNode );
                _graph_nodes.insert( new_node );
                
                // Connect the task with its parent task, if exists
                if( current_node != NULL ) {
                    Node::connect_nodes( current_node, new_node, Nesting );
                }
            }
            else if ( e.getKey( ) == user_funct_location )
            {   // A user function has been called
                int64_t func_id = e.getValue( );
                if( func_id != 0 && _funct_id_to_decl_map.find( func_id ) == _funct_id_to_decl_map.end( ) ) {
                    std::string description = iD->getValueDescription( user_funct_location, func_id );
                    int pos2 = description.find_first_of( "(" );
                    int pos1 = description.find_last_of ( " ", pos2 );
                    _funct_id_to_decl_map[ func_id ] = '\"' + description.substr( pos1+1, pos2-pos1-1 ) + '\"';
                }
            }
            else if ( e.getKey( ) == dependence )
            {  // A dependence occurs

                // Get the identifiers of the sender and the receiver
                int64_t sender_wd_id = (int64_t) ( ( e.getValue( ) >> 32) & 0xFFFFFFFF );
                int64_t receiver_wd_id = (int64_t) ( e.getValue( ) & 0xFFFFFFFF );
                
                // Get the type of dependence
                e = events[++i];
                assert( e.getKey( ) == dep_direction );
                DependencyType dep_type;
                unsigned dep_value = e.getValue( );
                switch( dep_value )
                {
                    case 1:     dep_type = True;        break;
                    case 2:     dep_type = Anti;        break;
                    case 3:     dep_type = Output;      break;
                    case 4:     dep_type = d_Output;    break;
                    case 5:     dep_type = Output_d;    break;
                    default:    { printf( "Unexpected type dependency %d. "
                                          "Not printing any edge for it in the Task Dependency graph\n", 
                                          dep_value );
                                  return; }
                }
                
                // Create the relation between the sender and the receiver
                Node* sender = find_node_from_wd_id( sender_wd_id );
                Node* receiver = find_node_from_wd_id( receiver_wd_id );
                assert( ( sender != NULL ) && ( receiver != NULL ) );
                Node::connect_nodes( sender, receiver, Dependency, dep_type );
            }
            else if ( e.getKey( ) == taskwait )
            {   // A taskwait occurs
                // Synchronize all previous nodes created by the same task that have not been yet synchronized
                Node* new_node = new Node( taskwait_id--, 0, TaskwaitNode );
                for( std::set<Node*>::iterator it = _graph_nodes.begin( ); it != _graph_nodes.end( ); ++it )
                {
                    if( (*it)->is_task( ) ) {
                        std::set<Edge*> entries = (*it)->get_entries( );
                        if( entries.empty( ) && ( current_node == NULL ) && !(*it)->is_synchronized( ) ) {
                            Node::connect_nodes( *it, new_node, Synchronization );
                        } else {
                            for( std::set<Edge*>::iterator edge = entries.begin( ); edge != entries.end( ); ++edge )
                            {
                                // The node is a child of 'current_wd_id' and
                                // it has no synchronization, synchronize it here
                                if( ( (*edge)->get_source( ) == current_node ) && 
                                    ( !(*it)->is_synchronized( ) ) )
                                {
                                    Node::connect_nodes( *it, new_node, Synchronization );
                                }
                            }
                        }
                    }
                }
                _graph_nodes.insert( new_node );
            }
        }
    }

    void threadStart( BaseThread &thread ) {}

    void threadFinish ( BaseThread &thread ) {}

#endif

};

namespace ext {
    
    class InstrumentationNewGraphInstrumentationPlugin : public Plugin {
    public:
        InstrumentationNewGraphInstrumentationPlugin () : Plugin("Instrumentation which print the graph to a dot file.",1) {}
        ~InstrumentationNewGraphInstrumentationPlugin () {}
        
        void config( Config &cfg ) {}
        
        void init ()
        {
            sys.setInstrumentation( new InstrumentationNewGraphInstrumentation() );
        }
    };
    
} // namespace ext

} // namespace nanos

DECLARE_PLUGIN("intrumentation-new-graph",nanos::ext::InstrumentationNewGraphInstrumentationPlugin);
