#include "graph_utils_new.hpp"

#include "instrumentation.hpp"
#include "instrumentationcontext_decl.hpp"
#include "os.hpp"
#include "plugin.hpp"
#include "smpdd.hpp"
#include "system.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

namespace {
    nanos::Lock lock;
}

namespace nanos {

const int64_t concurrent_min_id = 1000000;
static unsigned int cluster_id = 1;

class InstrumentationNewGraphInstrumentation: public Instrumentation
{
    private:
    std::set<Node*> _graph_nodes;                           /*!< relation between a wd id and its node in the graph */
    std::map<int64_t, std::string> _funct_id_to_decl_map;   /*!< relation between a task id and its name */
    double _time_avg;
    
    int64_t _next_tw_id;
    int64_t _next_conc_id;
    
    inline int64_t getMyWDId() {
        BaseThread *current_thread = getMyThreadSafe();
        if(current_thread == NULL) return 0;
        else if(current_thread->getCurrentWD() == NULL) return 0;
        return current_thread->getCurrentWD()->getId();
    }
    
    inline std::string print_node(Node* n, std::string indentation) {
        
        std::string node_attrs = "";
        // Get the label of the node
        {
            std::stringstream ss; ss << n->get_wd_id();
            if(n->is_taskwait()) {
                node_attrs += "label=\"Taskwait\", ";
            } else if(n->is_barrier()) {
                node_attrs += "label=\"Barrier\", ";
            } else if(n->is_concurrent()) {
                node_attrs += "label=\"Concurrent\", ";
            } else if(n->is_commutative()) {
                node_attrs += "label=\"Commutative\", ";
            }
        }
        
        // Get the style of the node
        node_attrs += "style=\"";
        node_attrs += (!n->is_task() ? "bold" : "filled");
        
        // Get the color of the node
        if(n->is_task()) {
            node_attrs += "\", color=\"black\", fillcolor=\"" + wd_to_color_hash(n->get_funct_id());
        }
        
        // Get the size of the node
        if(_time_avg == 0.0) {
            node_attrs += "\", width=\"1\", height=\"1\"";
        }
        else {
            double size = std::max(0.01, (double)n->get_total_time() / _time_avg);
            std::stringstream ss; ss << size;
            node_attrs += "\", width=\"" + ss.str() + "\", height=\"" + ss.str() + "\"";
        }
        
        // Build and return the whole node info
        std::stringstream ss; ss << n->get_wd_id();
        return std::string(indentation + ss.str()) + "[" + node_attrs + "];\n";
    }
    
    inline std::string print_edge(Edge* e, std::string indentation) {
        std::string edge_attrs = "style=\"";
        
        // Compute the style of the edge
        edge_attrs += ((!e->is_dependency() || 
                        e->is_true_dependency()) ? "solid" 
                                                 : (e->is_anti_dependency() ? "dashed" 
                                                                            : "dotted"));
        
        // Compute the color of the edge
        edge_attrs += "\", color=\"";
        edge_attrs += (e->is_nesting() ? "gray47" 
                                       : "black");
        edge_attrs += "\"";
        
        // Print the edge
        std::stringstream sss; sss << e->get_source()->get_wd_id();
        std::stringstream sst; sst << e->get_target()->get_wd_id();
        return std::string(indentation + sss.str() + " -> " + sst.str() + "[" + edge_attrs + "];\n");
    }
    
    inline std::string print_nested_nodes(Node* n, std::string indentation) {
        std::string nested_nodes_info = "";
        // Find all nodes which parent is 'n' and the edges connecting them is a 'Nesting' edge
        for(std::set<Node*>::iterator it = _graph_nodes.begin(); it != _graph_nodes.end(); ++it) 
        {
            std::vector<Edge*> entries = (*it)->get_entries();
            for(std::vector<Edge*>::iterator it2 = entries.begin(); it2 != entries.end(); ++it2)
            {
                if(((*it2)->get_source() == n) && (*it2)->is_nesting())
                {   // This is a nested relation!
                    nested_nodes_info += print_node(*it, indentation)
                                       + print_edge(*it2, indentation);
                    // Call recursively for nodes nested to the current node
                    print_nested_nodes(*it, std::string(indentation + "  "));
                }
            }
        }
        return nested_nodes_info;
    }
    
    inline std::string print_edges_legend() {
        std::stringstream ss; 
        lock.acquire();
            ss << cluster_id++;
        lock.release();
        std::string edges_legend = "";
        edges_legend += "  subgraph cluster_" + ss.str() + " {\n";
        edges_legend += "    label=\"Edge types:\"; style=\"rounded\"; rankdir=\"TB\";\n";
        edges_legend += "    subgraph {\n";
        edges_legend += "      rank=same\n";
        edges_legend += "      \"solid gray line\"[label=\"\", color=\"white\", shape=\"point\"];\n";
        edges_legend += "      \"Nested task\"[color=\"white\", margin=\"0.0,0.0\"];\n";
        edges_legend += "      \"solid gray line\"->\"Nested task\"[minlen=2.0, color=gray47];\n";
        edges_legend += "    }\n";
        edges_legend += "    subgraph {\n";
        edges_legend += "      rank=same;\n";
        edges_legend += "      \"solid black line\"[label=\"\", color=\"white\", shape=\"point\"];\n";
        edges_legend += "      \"True dependence \\n Taskwait | Barrier\"[color=\"white\", margin=\"0.0,0.0\"];\n";
        edges_legend += "      \"solid black line\"->\"True dependence \\n Taskwait | Barrier\"[minlen=2.0];\n";
        edges_legend += "    }\n";
        edges_legend += "    subgraph {\n";
        edges_legend += "      rank=same;\n";
        edges_legend += "      \"dashed line\"[label=\"\", color=\"white\", shape=\"point\"];\n";
        edges_legend += "      \"Anti-dependence\"[color=\"white\", margin=\"0.0,0.0\"];\n";
        edges_legend += "      \"dashed line\"->\"Anti-dependence\"[style=\"dashed\", minlen=2.0];\n";
        edges_legend += "    }\n";
        edges_legend += "    subgraph {\n";
        edges_legend += "      rank=same;\n";
        edges_legend += "      \"dotted line\"[label=\"\", color=\"white\", shape=\"point\"];\n";
        edges_legend += "      \"Output dependence\"[color=\"white\", margin=\"0.0,0.0\"];\n";
        edges_legend += "      \"dotted line\"->\"Output dependence\"[style=\"dotted\", minlen=2.0];\n";
        edges_legend += "    }\n";
        edges_legend += "    \"solid gray line\"->\"solid black line\"[style=\"invis\"];\n";
        edges_legend += "    \"solid black line\"->\"dashed line\"[style=\"invis\"];\n";
        edges_legend += "    \"dashed line\"->\"dotted line\"[style=\"invis\"];\n";
        edges_legend += "  }\n";
        return edges_legend;
    }
    
    inline std::string print_nodes_legend() {
        std::stringstream ssc; 
        lock.acquire();
        ssc << cluster_id++;
        lock.release();
        std::string nodes_legend = "";
        nodes_legend += "  subgraph cluster_" + ssc.str() + " {\n";
        nodes_legend += "    label=\"User functions:\"; style=\"rounded\"; rankdir=\"TB\";\n";
        
        int id = 1;
        std::set<std::string> printed_funcs;
        for(std::map<int64_t, std::string>::iterator it = _funct_id_to_decl_map.begin(); it != _funct_id_to_decl_map.end() ; ++it)
        {
            if(printed_funcs.find(it->second) == printed_funcs.end())
            {
                printed_funcs.insert(it->second);
                
                nodes_legend += "    subgraph {\n";
                nodes_legend += "      rank=same;\n";
                // Print the transparent node with the name of the function
                nodes_legend += "      " + _funct_id_to_decl_map[it->first] + "[color=\"white\", margin=\"0.0,0.0\"];\n";
                // Print one node for each function id that has the same name as the current function name
                int last_id = 0;
                for(std::map<int64_t, std::string>::iterator it2 = _funct_id_to_decl_map.begin(); 
                     it2 != _funct_id_to_decl_map.end(); ++it2)
                {
                    if(it2->second == it->second)
                    {
                        std::stringstream ssid; ssid << id;
                        nodes_legend += "      0" + ssid.str() + "[label=\"\",  width=0.3, height=0.3, shape=box, "
                                      + "fillcolor=" + wd_to_color_hash(it2->first) + ", style=filled];\n";
                        if(last_id != 0) {
                            std::stringstream ss2; ss2 << last_id;
                            nodes_legend += "      0" + ss2.str() + " -> 0" + ssid.str() + "[style=\"invis\"];\n";
                        }
                        last_id = id;
                        ++id;
                    }
                }
                // Print the edge between the last function id node and the name of the function
                std::stringstream sslid; sslid << last_id;
                nodes_legend += "      0" + sslid.str() + "->" + it->second + "[style=\"invis\"];\n";
                nodes_legend += "    }\n";
            }
        }
    
        // We want the pairs of <task_set, task_name> to be shown vertically
        // To achieve it, we create edges between each two consecutive pair subgraph
        std::set<std::string>::iterator it2;
        for(std::set<std::string>::iterator it = printed_funcs.begin(); it != printed_funcs.end(); ++it)
        {
            it2 = it; it2++;
            if(it2 != printed_funcs.end())
                nodes_legend += "    " + *it + " -> " + *it2 + "[style=\"invis\"];\n";
        }
        
        // Close the node legend subgraph
        nodes_legend += "  }\n";
        return nodes_legend;
    }
    
    inline Node* find_node_from_wd_id(int64_t wd_id) {
        Node* result = NULL;
        for(std::set<Node*>::iterator it = _graph_nodes.begin(); it != _graph_nodes.end(); ++it) {
            if((*it)->get_wd_id() == wd_id) {
                result = *it;
                break;
            }
        }
        return result;
    }
    
    inline std::string print_node_and_its_nested(Node* n, std::string indentation) {
        std::string result = "";
        // Print the current node
        std::string node_info = print_node(n, indentation);
        n->set_printed();
        // Print all nested nodes
        std::string nested_nodes_info = print_nested_nodes(n, /*indentation*/"    ");
        
        if(nested_nodes_info.empty()) {
            result = node_info;
        } else {
            // We want all nodes nested in a task to be printed horizontally
            result += "  subgraph {\n";
            result += "    rank=\"same\"; style=\"rounded\";\n";
            result += node_info;
            result += nested_nodes_info;
            result += "  }\n";
        }
        return result;
    }
    
    inline int get_cluster_id(std::map<int, int>& node_to_cluster, int cluster_inner_node_id, 
                              std::vector<Edge*> edges_to_from_cluster, bool cluster_is_source)
    {
        int current_cluster_id;
        std::map<int, int>::const_iterator cluster_inner_node_it = node_to_cluster.find(cluster_inner_node_id);
        if(cluster_inner_node_it == node_to_cluster.end())
        {
            lock.acquire();
            current_cluster_id = cluster_id++;
            lock.release();
        }
        else
        {
            current_cluster_id = cluster_inner_node_it->second;
            
            // Insert in the map the relation between all brothers of cluster_inner_node_id and the new cluster id
            if(cluster_is_source)
                for(std::vector<Edge*>::iterator e = edges_to_from_cluster.begin(); e != edges_to_from_cluster.end(); ++e)
                    node_to_cluster[(*e)->get_source()->get_wd_id()] = current_cluster_id;
            else
                for(std::vector<Edge*>::iterator e = edges_to_from_cluster.begin(); e != edges_to_from_cluster.end(); ++e)
                    node_to_cluster[(*e)->get_target()->get_wd_id()] = current_cluster_id;
        }
        
        return current_cluster_id;
    }
    
    inline std::string print_clustered_subgraph(int64_t current_wd, bool cluster_is_source, bool cluster_is_concurrent,
                                                 const std::vector<Edge*>& cluster_edges,
                                                 std::map<int, int>& node_to_cluster, std::vector<Edge*> cluster_exits) {
        std::string result = "";
        // Get the identifier of the cluster if it has been previously created or create a new identifier otherwise
        int cluster_inner_node_id = (cluster_is_source ? cluster_edges[0]->get_source()->get_wd_id() : 
                                                          cluster_edges[0]->get_target()->get_wd_id());
        int current_cluster_id = get_cluster_id(node_to_cluster, cluster_inner_node_id, cluster_edges, cluster_is_source);
        std::stringstream ss; ss << current_cluster_id;
        
        // Print all nodes that are concurrent|commutative with the current node inside the same subgraph
        result += "  subgraph cluster_" + ss.str() + "{\n";
        result += "    rank=\"same\"; style=\"rounded\"; ";
        if(cluster_is_concurrent)
            result += "label=\"Concurrent\"; \n";
        else
            result += "label=\"Commutative\"; \n";
        
        if(cluster_is_source) {
            for(std::vector<Edge*>::const_iterator e = cluster_edges.begin(); e != cluster_edges.end(); ++e)
                result += print_node_and_its_nested((*e)->get_source(), /*indentation*/"    ");
        } else {
            for(std::vector<Edge*>::const_iterator e = cluster_edges.begin(); e != cluster_edges.end(); ++e)
                result += print_node_and_its_nested((*e)->get_target(), /*indentation*/"    ");
        }
        result += "  }\n";
        
        for(std::vector<Edge*>::iterator it = cluster_exits.begin(); it != cluster_exits.end(); ++it)
        {
            std::vector<Edge*> actual_exits;
            if((*it)->get_target()->is_commutative() || (*it)->get_target()->is_concurrent())
                actual_exits = (*it)->get_target()->get_exits();
            else
                actual_exits.push_back(*it);
            for(std::vector<Edge*>::iterator e = actual_exits.begin(); e != actual_exits.end(); ++e) {
                std::stringstream sss; sss << current_wd;
                std::stringstream sst; sst << (*e)->get_target()->get_wd_id();
                std::stringstream ssct; ssct << get_cluster_id(node_to_cluster, (*e)->get_target()->get_wd_id(), 
                                                               actual_exits, /*cluster_is_source*/false);
                result += "  " + sss.str() + " -> " + sst.str() 
                        + "[ltail=\"cluster_" + ss.str() + "\", lhead=\"" + ssct.str() + "\", style=\"solid\", color=\"black\"];\n";
            }
        }
        
        return result;
    }
    
    inline std::string print_full_graph(std::string partial_file_name) {
        // Generate the name of the dot file from the name of the binary
        std::string result = partial_file_name + "_full";
        std::string file_name = result + ".dot";
        
        // Open the file and start the graph
        std::ofstream dot_file;
        dot_file.open(file_name.c_str());
        if(!dot_file.is_open())
            exit(EXIT_FAILURE);
        
        // Compute the time average to print the nodes size accordingly
        for(std::set<Node*>::iterator it = _graph_nodes.begin(); it != _graph_nodes.end(); ++it) {
            _time_avg += (*it)->get_total_time();
        }
        _time_avg /= _graph_nodes.size();
        
        // Print the graph
        std::map<int, int> node_to_cluster;
        dot_file << "digraph {\n";
            // Print attributes of the graph
            dot_file << "  graph[compound=true];\n";
            // Print the graph nodes
            for(std::set<Node*>::iterator it = _graph_nodes.begin(); it != _graph_nodes.end(); ++it)
            {
                if((*it)->is_printed())
                    continue;
                
                if((*it)->is_concurrent() || (*it)->is_commutative()) {
                    (*it)->set_printed();
                    continue;
                }
                
                std::vector<Edge*> entries = (*it)->get_entries();
                std::vector<Edge*> exits = (*it)->get_exits();
                if((entries.size() == 1) && (entries[0]->is_concurrent_dep() || entries[0]->is_commutative_dep()) && 
                   (exits.size() == 1) && (exits[0]->is_concurrent_dep() || exits[0]->is_commutative_dep()))
                {
                    /* This happens when we treat either T1 or T2, whichever is the first in _graph_nodes:
                     *       C
                     *     /   \
                     *   T1     T2      -> All concurrent nodes from the same task are printed inside
                     *     \   /        -> And the edges may be collapsed in one only entry edge and one only exit edge
                     *       C
                     */
                    std::vector<Edge*> edges_to_clustered_tasks = entries[0]->get_source()->get_exits();
                    dot_file << print_clustered_subgraph((*it)->get_wd_id(), /*cluster_is_source*/ false, 
                                                          /*cluster_is_concurrent*/ entries[0]->is_concurrent_dep(),
                                                          edges_to_clustered_tasks, node_to_cluster, exits);
                } else if((exits.size() == 1) && (exits[0]->is_concurrent_dep() || exits[0]->is_commutative_dep())) {
                    /* This happens when:
                     *    ...               -> Tasks may have some entry, but it is not a concurrent|commutative node
                     * T1     T2            -> T1 and T2 do not have any previous dependency
                     *   \   /                 so the first case of the IfElseStatement never occurs
                     *     C
                     */
                    std::vector<Edge*> edges_from_clustered_tasks = exits[0]->get_target()->get_entries();
                    dot_file << print_clustered_subgraph((*it)->get_wd_id(), /*cluster_is_source*/ true, 
                                                          /*cluster_is_concurrent*/ exits[0]->is_concurrent_dep(),
                                                          edges_from_clustered_tasks, node_to_cluster, exits);
                } else {
                    // Print the node and its nested nodes
                    dot_file << print_node_and_its_nested(*it, /*indentation*/"  ");
                    
                    // Print the exit edges (outside the rank, so they are displayed top-bottom)
                    std::set<Node*> nodes_in_same_cluster_to_avoid;
                    for(std::vector<Edge*>::iterator edge = exits.begin(); edge != exits.end(); ++edge) {
                        if(!(*edge)->is_nesting()) 
                        {   // nesting edges have been printed previously in 'print_nested_nodes'
                            if(!(*edge)->get_target()->is_concurrent() && !(*edge)->get_target()->is_commutative() && 
                                ((*edge)->get_target()->get_exits().empty() || 
                                  (!(*edge)->get_target()->get_exits()[0]->get_target()->is_concurrent() && 
                                    !(*edge)->get_target()->get_exits()[0]->get_target()->is_commutative())))
                            {
                                dot_file << print_edge(*edge, /*indentation*/"  ");
                            } 
                            else 
                            {
                                if(nodes_in_same_cluster_to_avoid.find((*edge)->get_target()) != nodes_in_same_cluster_to_avoid.end())
                                    continue;
                                
                                if((*edge)->get_target()->is_concurrent() || (*edge)->get_target()->is_commutative()) {
                                    /*     n
                                     *     |
                                     *     C
                                     *   /   \
                                     * ...   ...
                                     */
                                    std::stringstream ssc;
                                    if(node_to_cluster.find((*edge)->get_target()->get_exits()[0]->get_target()->get_wd_id()) != node_to_cluster.end()) {
                                        // Get the identifier of the cluster that has already been printed
                                        ssc << node_to_cluster[(*edge)->get_target()->get_exits()[0]->get_target()->get_wd_id()];
                                    } else {
                                        // Otherwise, we assign a cluster id for the new cluster that will be created, so we can link it now
                                        std::vector<Edge*> conc_or_comm_exits = (*edge)->get_target()->get_exits();
                                        int current_cluster_id;
                                        lock.acquire();
                                        current_cluster_id = cluster_id++;
                                        lock.release();
                                        for(std::vector<Edge*>::iterator e = conc_or_comm_exits.begin(); e != conc_or_comm_exits.end(); ++e) {
                                            node_to_cluster[(*e)->get_target()->get_wd_id()] = current_cluster_id;
                                        }
                                        ssc << current_cluster_id;
                                    }
                                    // Print the node in the dot file_name                                        
                                    std::stringstream sss; sss << (*it)->get_wd_id();
                                    std::stringstream sst; sst << (*edge)->get_target()->get_exits()[0]->get_target()->get_wd_id();
                                    dot_file << "  " << sss.str() + " -> " + sst.str() + "[lhead=\"cluster_" + ssc.str() + "\", style=\"solid\", color=\"black\"];\n";
                                    // The rest of nodes in the same cluster must not be connected
                                    std::vector<Edge*> conc_or_comm_exits = (*edge)->get_target()->get_exits();
                                    for(std::vector<Edge*>::iterator e = conc_or_comm_exits.begin(); e != conc_or_comm_exits.end(); ++e) {
                                        nodes_in_same_cluster_to_avoid.insert((*e)->get_target());
                                    }
                                } else {
                                    /*     n
                                     *   /   \    \
                                     *  T1   T2   Tn     -> Note that not all children tasks must be concurrent!
                                     *   \   /
                                     *     C
                                     */
                                    std::stringstream ssc;
                                    if(node_to_cluster.find((*edge)->get_target()->get_wd_id()) != node_to_cluster.end()) {
                                        // Get the identifier of the cluster that has already been printed
                                        ssc << node_to_cluster[(*edge)->get_target()->get_wd_id()];
                                    } else {
                                        // Otherwise, we assign a cluster id for the new cluster that will be created, so we can link it now
                                        int current_cluster_id;
                                        lock.acquire();
                                        current_cluster_id = cluster_id++;
                                        lock.release();
                                        for(std::vector<Edge*>::iterator e = exits.begin(); e != exits.end(); ++e) {
                                            if(((*e)->get_target()->get_exits().size() == 1) && 
                                                ((*e)->get_target()->get_exits()[0]->get_target()->is_concurrent() || 
                                                  (*e)->get_target()->get_exits()[0]->get_target()->is_commutative())) 
                                            {
                                                node_to_cluster[(*e)->get_target()->get_wd_id()] = current_cluster_id;
                                            }
                                        }
                                        ssc << current_cluster_id;
                                    }
                                    // Print the node in the dot file
                                    std::stringstream sss; sss << (*it)->get_wd_id();
                                    std::stringstream sst; sst << (*edge)->get_target()->get_wd_id();
                                    dot_file << "  " << sss.str() + " -> " + sst.str() + "[lhead=\"cluster_" + ssc.str() + "\", style=\"solid\", color=\"black\"];\n";
                                    // The rest of nodes in the same cluster must not be connected
                                    for(std::vector<Edge*>::iterator e = exits.begin(); e != exits.end(); ++e) {
                                        if(((*e)->get_target()->get_exits().size() == 1) && 
                                            ((*e)->get_target()->get_exits()[0]->get_target()->is_commutative() || 
                                              (*e)->get_target()->get_exits()[0]->get_target()->is_concurrent()))
                                        {
                                            nodes_in_same_cluster_to_avoid.insert((*e)->get_target());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Print the legends
            dot_file << print_nodes_legend();
            dot_file << print_edges_legend();
        dot_file << "}";
        
        std::cerr << "Task Dependency Graph printed to file '" << file_name << "' in DOT format" << std::endl;
        return result;
    }
    
#ifndef NANOS_INSTRUMENTATION_ENABLED
    public:
    // constructor
    InstrumentationNewGraphInstrumentation() : Instrumentation(),
                                               _graph_nodes(), _funct_id_to_decl_map(), 
                                               _time_avg(0.0), _next_tw_id(0), _next_conc_id(0)
    {}
    
    // destructor
    ~InstrumentationNewGraphInstrumentation() {}

    // low-level instrumentation interface (mandatory functions)
    void initialize(void) {}
    void finalize(void) {}
    void disable(void) {}
    void enable(void) {}
    void addResumeTask(WorkDescriptor &w) {}
    void addSuspendTask(WorkDescriptor &w, bool last) {}
    void addEventList (unsigned int count, Event *events) {}
    void threadStart(BaseThread &thread) {}
    void threadFinish (BaseThread &thread) {}
#else
    public:
    // constructor
    InstrumentationNewGraphInstrumentation() : Instrumentation(*new InstrumentationContextDisabled()),
                                               _graph_nodes(), _funct_id_to_decl_map(), 
                                               _time_avg(0.0), _next_tw_id(0), _next_conc_id(0)
    {}
    
    // destructor
    ~InstrumentationNewGraphInstrumentation () {}

    // low-level instrumentation interface (mandatory functions)
    void initialize(void) 
    {}
    
    void finalize(void)
    {
        // So far, taskwaits have been synchronized with the tasks created previously
        // But those tasks created after a given taskwait have not been connected to the taskwait
        // Note: The wd of a taskwait is always a negative number!
        for(std::set<Node*>::iterator it = _graph_nodes.begin(); it != _graph_nodes.end(); ++it) {
            if(!(*it)->is_previous_synchronized()) {
                // The task has no parent, look for a taskwait|barrier suitable to be its parent
                int64_t wd_id = (*it)->get_wd_id() - (((*it)->is_concurrent() || (*it)->is_commutative()) ? concurrent_min_id : 0);
                Node* it_parent = (*it)->get_parent_task();
                Node* last_taskwait_sync = NULL;
                for(std::set<Node*>::iterator it2 = _graph_nodes.begin(); it2 != _graph_nodes.end(); ++it2) {
                    if((it_parent == (*it2)->get_parent_task()) &&                  // The two nodes are in the same region
                        (!(*it2)->is_task()) &&                                      // The potential last sync is a taskwait|barrier|concurrent
                        (std::abs(wd_id) > std::abs((*it2)->get_wd_id()))) {    // The potential last sync was created before
                        if((last_taskwait_sync == NULL) || 
                            (last_taskwait_sync->get_wd_id() > (*it2)->get_wd_id())) {
                            // From all suitable previous syncs. we want the one created the latest
                            last_taskwait_sync = *it2;
                        }
                    }
                }
                if(last_taskwait_sync != NULL) {
                    Node::connect_nodes(last_taskwait_sync, *it, Synchronization);
                }
            }
        }
        
        // Get partial name of the file that will contain the dot graph
        std::string unique_key_name;
        time_t t = time(NULL);
        struct tm* tmp = localtime(&t);
        if (tmp == NULL) {
            std::stringstream ss; ss << getpid();
            unique_key_name = ss.str();
        } else {
            char outstr[200];
            if (strftime(outstr, sizeof(outstr), "%s", tmp) == 0) {
                std::stringstream ss; ss << getpid();
                unique_key_name = ss.str();
            } else {
                outstr[199] = '\0';
                unique_key_name = std::string(outstr);
            }
        }
        
        std::string file_name = OS::getArg(0);
        size_t slash_pos = file_name.find_last_of("/");
        if(slash_pos != std::string::npos)
            file_name = file_name.substr(slash_pos+1, file_name.size()-slash_pos);
        file_name = file_name + "_" + unique_key_name;
        
        // Print the full graph
        std::string full_dot_name = print_full_graph(file_name);
        
        // Generate the PDF containing the graph
        std::string dot_to_pdf_command = "dot -Tpdf " + full_dot_name + ".dot -o " + full_dot_name + ".pdf";
        if ( system( dot_to_pdf_command.c_str( ) ) != 0 )
            warning( "Could not create the pdf file." );
        std::cerr << "Task Dependency Graph printed to file '" << full_dot_name << ".pdf' in PDF format" << std::endl;
        
        // TODO
        // Print the summarized graph
//         print_summarized_graph(file_name);
    }

    void disable(void) {}

    void enable(void) {}

    static double get_current_time()
    {
        struct timeval tv;
        gettimeofday(&tv,0);
        return ((double) tv.tv_sec*1000000L) + ((double)tv.tv_usec);
    }

    void addResumeTask(WorkDescriptor &w)
    {
        Node* n = find_node_from_wd_id(w.getId());
        if(n != NULL) {
            n->set_last_time(get_current_time());
        }
    }

    void addSuspendTask(WorkDescriptor &w, bool last)
    {
        Node* n = find_node_from_wd_id(w.getId());
        if(n != NULL) {
            double time = (double) get_current_time() - n->get_last_time();
            n->add_total_time(time);
        }
    }

    void addEventList(unsigned int count, Event *events)
    {
        InstrumentationDictionary *iD = getInstrumentationDictionary();
        static const nanos_event_key_t create_wd_id = iD->getEventKey("create-wd-id");
        static const nanos_event_key_t create_wd_ptr = iD->getEventKey("create-wd-ptr");
        static const nanos_event_key_t dependence = iD->getEventKey("dependence");
        static const nanos_event_key_t dep_direction = iD->getEventKey("dep-direction");
        static const nanos_event_key_t user_funct_location = iD->getEventKey("user-funct-location");
        static const nanos_event_key_t taskwait = iD->getEventKey("taskwait");
        
        // Get the node corresponding to the wd_id calling this function
        // This node won't exist if the calling wd corresponds to that of the master thread
        int64_t current_wd_id = getMyWDId();
        Node* current_parent = find_node_from_wd_id(current_wd_id);
        
        unsigned int i;
        for(i=0; i<count; i++) {
            Event &e = events[i];
            if(e.getKey() == create_wd_ptr)
            {  // A wd is submitted => create a new node
                
                // Get the identifier of the task function
                WorkDescriptor *wd = (WorkDescriptor *) e.getValue();
                int64_t funct_id = (int64_t) ((ext::SMPDD &)(wd->getActiveDevice())).getWorkFct();
                
                // Get the identifier of the wd
                e = events[--i];
                assert(e.getKey() == create_wd_id);
                int64_t wd_id = e.getValue();
                _next_tw_id = std::min(_next_tw_id, -wd_id);
                _next_conc_id = wd_id + 1;
                // Create the new node
                Node* new_node = new Node(wd_id, funct_id, TaskNode);
                _graph_nodes.insert(new_node);
                
                // Connect the task with its parent task, if exists
                if(current_parent != NULL) {
                    Node::connect_nodes(current_parent, new_node, Nesting);
                }
            }
            else if (e.getKey() == user_funct_location)
            {   // A user function has been called
                int64_t func_id = e.getValue();
                if(func_id != 0 && _funct_id_to_decl_map.find(func_id) == _funct_id_to_decl_map.end()) {
                    std::string description = iD->getValueDescription(user_funct_location, func_id);
                    int pos2 = description.find_first_of("(");
                    int pos1 = description.find_last_of (" ", pos2);
                    _funct_id_to_decl_map[ func_id ] = '\"' + description.substr(pos1+1, pos2-pos1-1) + '\"';
                }
            }
            else if (e.getKey() == dependence)
            {  // A dependence occurs

                // Get the identifiers of the sender and the receiver
                int64_t sender_wd_id = (int64_t) ((e.getValue() >> 32) & 0xFFFFFFFF);
                int64_t receiver_wd_id = (int64_t) (e.getValue() & 0xFFFFFFFF);
                
                // Get the type of dependence
                e = events[++i];
                assert(e.getKey() == dep_direction);
                DependencyType dep_type;
                unsigned dep_value = e.getValue();
                switch(dep_value)
                {
                    case 1:     dep_type = True;
                                break;
                    case 2:     dep_type = Anti;
                                break;
                    case 3:     dep_type = Output;
                                break;
                    case 4:     dep_type = OutConcurrent;               // wd -> concurrent
                                receiver_wd_id += concurrent_min_id;
                                break;
                    case 5:     dep_type = InConcurrent;                // concurrent -> wd
                                sender_wd_id += concurrent_min_id;
                                break;
                    case 6:     dep_type = OutCommutative;
                                receiver_wd_id += concurrent_min_id;    // wd -> commutative
                                break;
                    case 7:     dep_type = InCommutative;
                                sender_wd_id += concurrent_min_id;      // commutative -> wd
                                break;
                    // FIXME Case 8 and 9 must have a different behavior
                    case 8:     dep_type = OutConcurrent;
                                sender_wd_id += concurrent_min_id;      // wd -> common
                                break;
                    case 9:     dep_type = InConcurrent;
                                sender_wd_id += concurrent_min_id;      // common -> wd
                                break;
                    default:    { std::cerr << "Unexpected type dependency " << dep_value << "."
                                            << "Not printing any edge for it in the Task Dependency graph\n"
                                            << std::endl;
                                  return; }
                }
                
                // Create the relation between the sender and the receiver
                Node* sender = find_node_from_wd_id(sender_wd_id);
                Node* receiver = find_node_from_wd_id(receiver_wd_id);
                if(sender == NULL || receiver == NULL) {
                    // Compute the type for the new node
                    NodeType nt = (((dep_type == InConcurrent) || (dep_type == OutConcurrent)) ? ConcurrentNode : CommutativeNode);
                    
                    // Create the new sender node, if necessary
                    if(sender == NULL) {
                        sender = new Node(sender_wd_id, 0, nt);
                        _graph_nodes.insert(sender);
                    }
                    
                    // Create the new receiver node, if necessary
                    if(receiver == NULL) {
                        receiver = new Node(receiver_wd_id, 0, nt);
                        _graph_nodes.insert(receiver);
                    }
                }
                Node::connect_nodes(sender, receiver, Dependency, dep_type);
            }
            else if (e.getKey() == taskwait)
            {   // A taskwait occurs
                // Synchronize all previous nodes created by the same task that have not been yet synchronized
                Node* new_node = new Node(_next_tw_id, -1, TaskwaitNode);
                --_next_tw_id;
                // First synchronize the tasks
                for(std::set<Node*>::iterator it = _graph_nodes.begin(); it != _graph_nodes.end(); ++it) {
                    // Synchronization nodes will be connected, if necessary, when 'finalize' is called
                    if(!(*it)->is_next_synchronized() && (*it)->is_task()) {
                        Node* parent_task = (*it)->get_parent_task();
                        if(current_parent == parent_task) {
                            Node::connect_nodes(*it, new_node, Synchronization);
                        }
                    }
                }
                _graph_nodes.insert(new_node);
            }
        }
    }

    void threadStart(BaseThread &thread) {}

    void threadFinish (BaseThread &thread) {}

#endif

};

namespace ext {
    
    class InstrumentationNewGraphInstrumentationPlugin : public Plugin {
    public:
        InstrumentationNewGraphInstrumentationPlugin () : Plugin("Instrumentation which print the graph to a dot file.",1) {}
        ~InstrumentationNewGraphInstrumentationPlugin () {}
        
        void config(Config &cfg) {}
        
        void init ()
        {
            sys.setInstrumentation(new InstrumentationNewGraphInstrumentation());
        }
    };
    
} // namespace ext

} // namespace nanos

DECLARE_PLUGIN("intrumentation-new-graph",nanos::ext::InstrumentationNewGraphInstrumentationPlugin);
