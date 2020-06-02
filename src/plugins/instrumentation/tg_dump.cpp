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

#include "tg_dump_utils.hpp"

#include "instrumentation.hpp"
#include "instrumentationcontext_decl.hpp"
#include "os.hpp"
#include "plugin.hpp"
#include "smpdd.hpp"
#include "system.hpp"
#include "papi.h"

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <math.h>
#include <queue>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

namespace {
    nanos::Lock lock;
}

namespace nanos {

enum { concurrent_min_id = 1000000 };
static unsigned int cluster_id = 1;

class InstrumentationTGDump: public Instrumentation
{
public:
    static std::list<std::string> _papi_event_args;
    static std::string _output_mode;
private:
    Node* _root;
    std::set<Node*> _graph_nodes;                           /*!< relation between a wd id and its node in the graph */
    Lock _graph_nodes_lock;
    std::map<int64_t, std::string> _funct_id_to_decl_map;   /*!< relation between a task id and its name */
    Lock _funct_id_to_decl_map_lock;
    double _min_time;
    double _total_time;
    double _min_diam;
    std::vector<int> _papi_event_codes;                     // Event codes to be tracked using PAPI
    bool _commutative_output_warning_printed = false;
    bool _commutative_task_warning_printed = false;

#ifdef NANOS_INSTRUMENTATION_ENABLED
    int64_t _next_tw_id;
    int64_t _next_conc_id;
#endif

    inline int64_t getMyWDId() {
        BaseThread *current_thread = getMyThreadSafe();
        if(current_thread == NULL) return 0;
        else if(current_thread->getCurrentWD() == NULL) return 0;
        return current_thread->getCurrentWD()->getId();
    }

    inline Node* find_node_from_wd_id(int64_t wd_id) const {
        Node* result = NULL;
        for(std::set<Node*>::const_iterator it = _graph_nodes.begin(); it != _graph_nodes.end(); ++it) {
            if((*it)->get_wd_id() == wd_id) {
                result = *it;
                break;
            }
        }
        return result;
    }

//====[DOT PRINTING]==========================================================================================================//

    inline std::string print_node_dot(Node* n, std::string indentation) {
        std::stringstream ss;

        // Create node label and open attribute braces
        ss << indentation << n->get_wd_id() << "[";

        // Get the label of the node
        if (n->is_taskwait()) {
            ss << "label=\"Taskwait\", ";
        } else if (n->is_barrier()) {
            ss << "label=\"Barrier\", ";
        } else {
            // Open html label
            ss << "label=<<table border=\"0\" cellspacing=\"5\" cellborder=\"0\">";

            // Output papi operation counters as html label
            std::vector<std::pair<int, long long> > perf_counters = n->get_perf_counters();
            std::vector<std::pair<int, long long> >::iterator it;
            for(it = perf_counters.begin(); it != perf_counters.end(); ++it) {
                char papi_event_name[PAPI_MAX_STR_LEN];
                int rc;

                if((rc = PAPI_event_code_to_name(it->first, papi_event_name)) != PAPI_OK) {
                    std::cerr << "Failed to get name for event id " << it->first << ". ";
                    std::cerr << "Papi error: (" << rc << ") - " << PAPI_strerror(rc) << ". ";
                    std::cerr << "The associated counter will not be emitted.\n";
                    continue;
                }

                // Output row for papi counter
                ss << "<tr>";
                ss << "<td>" << std::string(papi_event_name) <<"</td>";
                ss << "<td>" << it->second << "</td>";
                ss << "</tr>";
            }

            // Output execution time
            ss << "<tr>";
            ss << "<td>time</td>";
            ss << "<td>" << formatTime(n->get_total_time()) << "</td>";
            ss << "</tr>";

            // Close html label
            ss << "</table>>";
        }

        // Get the style of the node
        ss << "style=\"";
        ss << (!n->is_task() ? "bold" : "filled");
        ss << (n->is_task() && n->is_critical() ? ",bold\", shape=\"doublecircle" : "");    //Mark critical tasks as bold and filled

        // Get the color of the node
        if(n->is_task()) {
            std::string description = _funct_id_to_decl_map[n->get_funct_id()];
            ss << "\", color=\"black\", fillcolor=\"" << wd_to_color_hash(description);
        }

        // Set the size of the node
        ss << "\", width=\"1\", height=\"1\"";

        // Close the braces around the node attributes
        ss << "];\n";

        return ss.str();
    }

    inline std::string print_edge_dot(Edge* e, std::string indentation)
    {
        std::string edge_attrs = "style=\"";

        // Compute the style of the edge
        edge_attrs += ((!e->is_dependency() || e->is_true_dependency()) ? "solid" : (e->is_anti_dependency() ? "dashed" : "dotted"));

        //Mark the edges of the critical path as bold
        edge_attrs += ( (e->get_source()->is_critical() && e->get_target()->is_critical()) ? ",bold" : "" );

        // Compute the color of the edge
        edge_attrs += "\", color=";
        edge_attrs += (e->is_nesting() ? "\"gray47\"" : "\"black\"");

        edge_attrs += ", label=\"" + formatSize(e->get_data_size()) + "\"";

        // Print the edge
        std::stringstream sss; sss << e->get_source()->get_wd_id();
        std::stringstream sst; sst << e->get_target()->get_wd_id();
        return std::string(indentation + sss.str() + " -> " + sst.str() + " [" + edge_attrs + "];\n");
    }

    inline std::string print_nested_nodes_dot(Node* n, std::string indentation) {
        std::string nested_nodes_info = "";
        std::vector<Edge*> const& exits = n->get_exits();
        std::vector<Edge*> nested_exits;
        for(std::vector<Edge*>::const_iterator it = exits.begin(); it != exits.end(); ++it)
        {
            if ((*it)->is_nesting())
            {
                Node* t = (*it)->get_target();
                nested_nodes_info += print_node_dot(t, indentation) + print_edge_dot(*it, indentation);
                // Call recursively for nodes nested to the current node
                print_nested_nodes_dot(t, std::string(indentation + "  "));
            }
        }
        return nested_nodes_info;
    }

    inline std::string print_edges_legend_dot() {
        std::stringstream ss;
        lock.acquire();
            ss << cluster_id++;
        lock.release();
        std::string edges_legend = "";

        // Avoid printing an empty legend
        if (!used_edge_types[0] && !used_edge_types[1] && !used_edge_types[2]
            && !used_edge_types[3] && !used_edge_types[4])
            return "";

        // Open the subgraph containing the edge's legend
        edges_legend += "  subgraph cluster_" + ss.str() + " {\n";
        edges_legend += "    label=\"Edge types:\"; style=\"rounded\";\n";

        // Print the table with the used edge types
        edges_legend += "    edges_table [label=<<table border=\"0\" cellspacing=\"10\" cellborder=\"0\">\n";
        if (used_edge_types[0])
        {
            edges_legend += "      <tr>\n";
            edges_legend += "        <td width=\"15px\" border=\"0\">&#10141;</td>\n";
            edges_legend += "        <td>True dependence | Taskwait | Barrier</td>\n";
            edges_legend += "      </tr>\n";
        }
        if (used_edge_types[1])
        {
            edges_legend += "      <tr>\n";
            edges_legend += "        <td width=\"15px\" border=\"0\">&#8674;</td>\n";
            edges_legend += "        <td>Anti-dependence</td>\n";
            edges_legend += "      </tr>\n";
        }
        if (used_edge_types[2])
        {
            edges_legend += "      <tr>\n";
            edges_legend += "        <td width=\"15px\" border=\"0\">&#10513;</td>\n";
            edges_legend += "        <td>Output dependence</td>\n";
            edges_legend += "      </tr>\n";
        }
        if (used_edge_types[3])
        {
            edges_legend += "      <tr>\n";
            edges_legend += "        <td width=\"15px\" border=\"0\"><font color=\"gray47\">&#10141;</font></td>\n";
            edges_legend += "        <td>Nested task</td>\n";
            edges_legend += "      </tr>\n";
        }
        if (used_edge_types[4])
        {
            edges_legend += "      <tr>\n";
            edges_legend += "        <td width=\"15px\" border=\"0\">&#10145;</td>\n";
            edges_legend += "        <td>Critical path</td>\n";
            edges_legend += "      </tr>\n";
        }
        edges_legend += "    </table>>]\n";

        // Close the node legend subgraph
        edges_legend += "  }\n";

        return edges_legend;
    }

    inline std::string print_nodes_legend_dot() {
        std::stringstream ssc;
        lock.acquire();
        ssc << cluster_id++;
        lock.release();

        std::string nodes_legend = "";

        // Avoid printing an empty legend
        if (_funct_id_to_decl_map.empty())
            return "";

        // Open the subgraph containing the node's legend
        nodes_legend += "  subgraph cluster_" + ssc.str() + " {\n";
        nodes_legend += "    label=\"User functions:\"; style=\"rounded\";\n";
        nodes_legend += "    funcs_table [label=<<table border=\"0\" cellspacing=\"10\" cellborder=\"0\">\n";
        std::set<std::string> printed_funcs;
        for (std::map<int64_t, std::string>::const_iterator it = _funct_id_to_decl_map.begin();
             it != _funct_id_to_decl_map.end() ; ++it)
        {
            std::string description = it->second;
            if (printed_funcs.find(description) == printed_funcs.end())
            {
                printed_funcs.insert(description);

                // Replace any '&' with the HTML entity '&amp;'
                std::string from = "&";
                std::string to = "&amp;";
                size_t start_pos = 0;
                std::string dot_description = description;
                while((start_pos = dot_description.find(from, start_pos)) != std::string::npos) {
                    dot_description.replace(start_pos, from.length(), to);
                    start_pos += to.length();
                }

                nodes_legend += "      <tr>\n";
                nodes_legend += "        <td bgcolor=\"" + wd_to_color_hash(description) + "\" width=\"15px\" border=\"1\"></td>\n";
                nodes_legend += "        <td>" + dot_description + "</td>\n";
                nodes_legend += "      </tr>\n";
            }
        }
        nodes_legend += "    </table>>]\n";

        // Close the node legend subgraph
        nodes_legend += "  }\n";
        return nodes_legend;
    }

    inline std::string print_node_and_its_nested_dot(Node* n, std::string indentation) {
        std::string result = "";
        // Print the current node
        std::string node_info = print_node_dot(n, indentation);
        n->set_printed();
        // Print all nested nodes
        std::string nested_nodes_info = print_nested_nodes_dot(n, /*indentation*/"    ");

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

    inline std::string print_full_graph_dot(std::string partial_file_name) {
        // Generate the name of the dot file from the name of the binary
        std::string result = partial_file_name + "_full";
        std::string file_name = result + ".dot";

        // Open the file and start the graph
        std::ofstream dot_file;
        dot_file.open(file_name.c_str());
        if(!dot_file.is_open())
            exit(EXIT_FAILURE);

        // Print the graph
        //std::map<int, int> node_to_cluster;
        dot_file << "digraph {\n";
        // Print attributes of the graph
        dot_file << "  graph[compound=true];\n";
        // Print the graph nodes
        std::queue<Node*> worklist;
        std::vector<Edge*> const &root_exits = _root->get_exits();
        for (std::vector<Edge*>::const_iterator it = root_exits.begin(); it != root_exits.end(); ++it)
            worklist.push((*it)->get_target());
        while (!worklist.empty())
        {
            Node* n = worklist.front();
            worklist.pop();

            std::vector<Edge*> const &exits = n->get_exits();
            if (n->is_printed())
                continue;
            if (n->is_commutative() || n->is_concurrent())
            {
                n->set_printed();
                for (std::vector<Edge*>::const_iterator it = exits.begin(); it != exits.end(); ++it)
                {
                    worklist.push((*it)->get_target());
                }
                continue;
            }

            // Check whether there is a block of commutative
            std::vector<Node*> out_commutatives;
            for (std::vector<Edge*>::const_iterator it = exits.begin(); it != exits.end(); ++it) {
                if ((*it)->is_commutative_dep())
                    out_commutatives.push_back((*it)->get_target());
            }

            // Print the current node and, if that is the case, its commutative nodes too
            if (out_commutatives.empty()) {
                // Print the node
                dot_file << print_node_and_its_nested_dot(n, /*indentation*/"  ");
                std::stringstream sss; sss << n->get_wd_id();

                // Print the relations with its children (or the children of the children if they are a virtual node)
                for (std::vector<Edge*>::const_iterator it = exits.begin(); it != exits.end(); ++it) {
                    if ((*it)->is_nesting()) continue;
                    Node* t = (*it)->get_target();

                    if (t->is_concurrent() || t->is_commutative()) {
                        std::vector<Edge*> const &it_exits = t->get_exits();
                        for (std::vector<Edge*>::const_iterator itt = it_exits.begin();
                             itt != it_exits.end(); ++itt) {
                           std::stringstream sst;
                           sst << (*itt)->get_target()->get_wd_id();
                           dot_file << "  " << sss.str() + " -> " + sst.str() + ";\n"; //TODO attributes?
                           worklist.push((*itt)->get_target());
                        }
                    } else {
                       dot_file << print_edge_dot( *it, /*indentation*/"  " );
                       worklist.push(t);
                    }
                }
            } else {
                // Since Graphviz does not allow boxes intersections and
                // multiple dependencies in a commutative clause may cause that situation,
                // we enclose all tasks in the same box and then print the dependencies individually
                // 1.- Get all nodes that must be in the commutative box
                std::vector<Node*> commutative_box;
                for (std::vector<Node*>::const_iterator it = out_commutatives.begin();
                     it != out_commutatives.end(); ++it) {
                    // Gather all siblings (parents of the out_commutative nodes)
                    std::vector<Edge*> const &it_entries = (*it)->get_entries();
                    for (std::vector<Edge*>::const_iterator itt = it_entries.begin();
                         itt != it_entries.end(); ++itt) {
                        commutative_box.push_back((*itt)->get_source());
                    }
                }
                // 2.- Print the subgraph with all sibling nodes
                std::stringstream ss; ss << cluster_id++;
                dot_file << "  subgraph cluster_" << ss.str() << "{\n";
                    dot_file << "    rank=\"same\"; style=\"rounded\"; label=\"Commutative\";\n";
                    for (std::vector<Node*>::iterator it = commutative_box.begin();
                         it != commutative_box.end(); ++it) {
                        dot_file << print_node_and_its_nested_dot(*it, /*indentation*/"    ");
                    }
                dot_file << "  }\n";
                // 3.- Print the edges connecting each sibling node with its corresponding real children
                for (std::vector<Node*>::iterator it = commutative_box.begin();
                     it != commutative_box.end(); ++it) {
                    std::stringstream sss; sss << (*it)->get_wd_id();
                    std::vector<Edge*> const &it_exits = (*it)->get_exits();
                    for (std::vector<Edge*>::const_iterator itt = it_exits.begin();
                         itt != it_exits.end(); ++itt) {
                        Node* t = (*itt)->get_target();
                        if ((*itt)->is_nesting()) continue;

                        if (t->is_commutative() || t->is_concurrent())
                        {   // If the exit is commutative|concurrent, get the exit of the exit
                            std::vector<Edge*> const &itt_exits = t->get_exits();
                            for (std::vector<Edge*>::const_iterator ittt = itt_exits.begin();
                                 ittt != itt_exits.end(); ++ittt) {
                                std::stringstream sst; sst << (*ittt)->get_target()->get_wd_id();
                                dot_file << "    " << sss.str() + " -> " + sst.str() + ";\n";
                                // Prepare the next iteration
                                worklist.push((*ittt)->get_target());
                            }
                        } else {
                            dot_file << print_edge_dot( *itt, /*indentation*/"  " );
                            // Prepare the next iteration
                            worklist.push(t);
                        }
                    }
                }
            }
        }

        // Print the legends
        dot_file << "  node [shape=plaintext];\n";
        dot_file << print_nodes_legend_dot();
        dot_file << print_edges_legend_dot();
        dot_file << "}";

        std::cerr << "Task Dependency Graph printed to file '" << file_name << "' in DOT format" << std::endl;
        return result;
    }

//====[JSON PRINTING]=========================================================================================================//

    //inline std::string print_graph_json(Node* root, std::string indent);

    inline std::string print_node_json(Node* n, std::string indent) {
        std::stringstream ss;

        ss << indent << "{\n";

        printJsonAttribute(indent + "  ", "id", n->get_wd_id(), ss);
        ss << ",\n";

        if (n->is_taskwait()) {
            printJsonAttribute(indent + "  ", "type", "Taskwait", ss);
        } else if (n->is_barrier()) {
            printJsonAttribute(indent + "  ", "type", "Barrier", ss);
        } else {
            printJsonAttribute(indent + "  ", "type", "Usertask", ss);
            ss << ",\n";

            std::string description = _funct_id_to_decl_map[n->get_funct_id()];
            std::vector<std::string> desc_sections;
            desc_sections.reserve(4);

            size_t end;
            while ((end = description.find_first_of("@")) != std::string::npos) {
                desc_sections.push_back(description.substr(0, end));
                description = description.substr(end + 1);
            }
            desc_sections.push_back(description);

            // If the format of the description is correct, print attributes
            if (desc_sections.size() == 4 && (desc_sections[3] == "LABEL" || desc_sections[3] == "FUNCTION")) {
                if (desc_sections[3] == "LABEL") {
                    printJsonAttribute(indent + "  ", "label", desc_sections[0], ss);
                } else {
                    printJsonNullAttribute(indent + "  ", "label", ss);
                }
                ss << ",\n";

                printJsonAttribute(indent + "  ", "file", desc_sections[1], ss);
                ss << ",\n";

                int line_number;
                std::istringstream(desc_sections[2]) >> line_number;
                printJsonAttribute(indent + "  ", "line", line_number, ss);
                ss << ",\n";
            }

            // Else, print the whole description
            else {
                printJsonAttribute(indent + "  ", "description", description, ss);
                ss << ",\n";
            }

            printJsonAttribute(indent + "  ", "critical", n->is_critical(), ss);
            ss << ",\n";

            printJsonAttribute(indent + "  ", "duration_us", n->get_total_time(), ss);
            ss << ",\n";

            // Output papi operation counts
            std::vector<std::pair<int, long long> > papi_counter_data = n->get_perf_counters();
            std::vector<std::pair<std::string, long long> > papi_counter_data_printable;

            papi_counter_data_printable.reserve(papi_counter_data.size());

            for(std::vector<std::pair<int, long long> >::iterator it = papi_counter_data.begin(); it != papi_counter_data.end(); ++it) {
                char papi_event_name[PAPI_MAX_STR_LEN];
                int rc;

                if((rc = PAPI_event_code_to_name(it->first, papi_event_name)) != PAPI_OK) {
                    std::cerr << "Failed to get name for event id " << it->first << ". ";
                    std::cerr << "Papi error: (" << rc << ") - " << PAPI_strerror(rc) << ". ";
                    std::cerr << "The associated counter will not be emitted.\n";
                    continue;
                }

                papi_counter_data_printable.push_back(std::make_pair(std::string(papi_event_name), it->second));
            }

            printJsonAttributeArray(indent + "  ", "papi_counters", papi_counter_data_printable, ss);
            ss << ",\n";

            // Get nested exits
            std::vector<Edge*> const& exits = n->get_exits();
            std::vector<Edge*> nested_exits;
            nested_exits.reserve(exits.size());

            for(std::vector<Edge*>::const_iterator it = exits.begin(); it != exits.end(); ++it) {
                if((*it)->is_nesting()) {
                    nested_exits.push_back(*it);
                }
            }

            // If we have nested exits, print subgraph
            if(nested_exits.size() != 0) {
                ss << indent + "  " << "\"subgraph\": {\n";
                ss << print_graph_objects_json(n, indent + "  " + "  ", true) << "\n";
                ss << indent + "  " << "}";
            } else {
                printJsonNullAttribute(indent + "  ", "subgraph", ss);
            }
        }

        ss << "\n" << indent << "}";
        return ss.str();
    }

    inline std::string print_edge_json(Edge* e, std::string indent)
    {
        std::stringstream ss;
        ss << indent << "{\n";

        printJsonAttribute(indent + "  ", "from", e->get_source()->get_wd_id(), ss);
        ss << ",\n";
        printJsonAttribute(indent + "  ", "to", e->get_target()->get_wd_id(), ss);
        ss << ",\n";

        // Print edge type
        std::string dep_type;
        dep_type = ((!e->is_dependency() || e->is_true_dependency()) ? "true" : (e->is_anti_dependency() ? "anti" : "output"));
        printJsonAttribute(indent + "  ", "dependence_type", dep_type, ss);
        ss << ",\n";

        printJsonAttribute(indent + "  ", "data_size", e->get_data_size(), ss);

        ss << "\n" << indent << "}";

        return ss.str();
    }

    inline std::string print_graph_objects_json(Node* root, std::string indent, bool nested_only = false) {
        std::stringstream oss;

        // Print the graph nodes
        std::queue<Node*> worklist;
        std::vector<Edge*> const &root_exits = root->get_exits();

        // Optionally only print nested nodes
        for (std::vector<Edge*>::const_iterator it = root_exits.begin(); it != root_exits.end(); ++it) {
            if(!nested_only || (*it)->is_nesting()) {
                worklist.push((*it)->get_target());
            }
        }

        std::vector<Node*> node_print_list;
        std::vector<Edge*> edge_print_list;

        while (!worklist.empty()) {
            Node* n = worklist.front();
            worklist.pop();

            if (n->is_printed()) {
                continue;
            }

            // If we aren't printing nested nodes, ignore nodes that are nested elsewhere
            if(!nested_only) {
                std::vector<Edge*> const &entries = n->get_entries();
                bool node_is_nested_somewhere = false;

                for (std::vector<Edge*>::const_iterator it = entries.begin(); it != entries.end(); ++it) {
                    if((*it)->is_nesting()) {
                        node_is_nested_somewhere = true;
                    }
                }

                if(node_is_nested_somewhere) {
                    continue;
                }
            }

            std::vector<Edge*> const &exits = n->get_exits();

            if (n->is_commutative() || n->is_concurrent()) {
                for (std::vector<Edge*>::const_iterator it = exits.begin(); it != exits.end(); ++it) {
                    worklist.push((*it)->get_target());
                }
                continue;
            }

            // Check whether there is a block of commutative
            std::vector<Node*> out_commutatives;
            for (std::vector<Edge*>::const_iterator it = exits.begin(); it != exits.end(); ++it) {
                if ((*it)->is_commutative_dep()) {
                    if(!_commutative_output_warning_printed) {
                        std::cerr << "Warning, encountered commutative output dependency.\n";
                        std::cerr << "When in JSON mode, commutative dependencies receieve no special handling.\n";
                        std::cerr << "This warning will only appear once.\n";
                        _commutative_output_warning_printed = true;
                    }
                }
            }

            // Print the node
            node_print_list.push_back(n);
            n->set_printed();
            std::stringstream sss; sss << n->get_wd_id();

            // Print the relations with its children (or the children of the children if they are a virtual node)
            for (std::vector<Edge*>::const_iterator it = exits.begin(); it != exits.end(); ++it) {
                if ((*it)->is_nesting()) continue;
                Node* t = (*it)->get_target();

                // TODO this section will still output stuff in dot format
                // it needs to be changed, but I don't really understand what is going on here.
                if (t->is_concurrent() || t->is_commutative()) {
                    if (!_commutative_task_warning_printed) {
                        std::cerr << "Warning, encountered commutative/concurrent task.\n";
                        std::cerr << "When in JSON mode, commutative/concurrent tasks receive no special handling.\n";
                        std::cerr << "This warning will only appear once.\n";
                        _commutative_task_warning_printed = true;
                    }
                }

                edge_print_list.push_back(*it);
                worklist.push(t);
            }
        }

        // Print nodes
        oss << indent << "\"nodes\": [\n";
        for (std::vector<Node*>::iterator it = node_print_list.begin(); it != node_print_list.end();) {
            oss << print_node_json(*it, indent + "  ");
            if (++it != node_print_list.end()) {
                oss << ",";
            }
            oss << "\n";
        }
        oss << indent << "],\n";

        // Print edges
        oss << indent << "\"edges\": [\n";
        for (std::vector<Edge*>::iterator it = edge_print_list.begin(); it != edge_print_list.end();) {
            oss << print_edge_json(*it, indent + "  ");
            if (++it != edge_print_list.end()) {
                oss << ",";
            }
            oss << "\n";
        }
        oss << indent << "]";

        return oss.str();
    }

    inline std::string print_full_graph_json(std::string partial_file_name) {
        // Generate the name of the dot file from the name of the binary
        std::string result = partial_file_name + "_full";
        std::string file_name = result + ".json";

        // Open the file and start the graph
        std::ofstream file;
        file.open(file_name.c_str());
        if(!file.is_open()) {
            exit(EXIT_FAILURE);
        }

        file << "{\n";
        file << print_graph_objects_json(_root, "  ") << "\n";
        file << "}\n";

        std::cerr << "Task Dependency Graph printed to file '" << file_name << "' as json" << std::endl;
        return result;
    }

#ifndef NANOS_INSTRUMENTATION_ENABLED
public:
    // constructor
    InstrumentationTGDump() : Instrumentation(),
                              _graph_nodes(), _funct_id_to_decl_map(),
                              _min_time(HUGE_VAL), _total_time(0.0), _min_diam(1.0)
    {}

    // destructor
    ~InstrumentationTGDump() {}

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
    InstrumentationTGDump() : Instrumentation(*new InstrumentationContextDisabled()),
                              _graph_nodes(), _funct_id_to_decl_map(),
                              _min_time(HUGE_VAL), _total_time(0.0), _min_diam(1.0),
                              _next_tw_id(0), _next_conc_id(0)
    {}

    // destructor
    ~InstrumentationTGDump () {}

    // low-level instrumentation interface (mandatory functions)
    void initialize(void)
    {
        int rc, test_event_set = PAPI_NULL;

        // Initialise PAPI
        if((rc = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) {
            std::cerr << "Failed to initialise PAPI library. ";
            std::cerr << "Papi error: (" << rc << ") - " << PAPI_strerror(rc) << ".\n";
            exit(1);
        }

        // Create a test event set
        if((rc = PAPI_create_eventset(&test_event_set)) != PAPI_OK) {
            std::cerr << "Failed to create test event set. ";
            std::cerr << "Papi error: (" << rc << ") - " << PAPI_strerror(rc) << ".\n";
            exit(1);
        }

        std::list<std::string>& papi_event_args = InstrumentationTGDump::_papi_event_args;
        std::list<std::string>::iterator it;

        // populate list of papi counter IDs
        for(it = papi_event_args.begin(); it != papi_event_args.end(); ++it) {
            int event_code;

            // Is this a valid name? If not, omit it
            if((rc = PAPI_event_name_to_code((*it).c_str(), &event_code)) != PAPI_OK) {
                std::cerr << "Failed to get event code for event '" << *it << "'. ";
                std::cerr << "Papi error: (" << rc << ") - " << PAPI_strerror(rc) << ". ";
                std::cerr << "This event will not be tracked.\n";
                continue;
            }

            // Can we add the id to an event list? If not, omit it
            if((rc = PAPI_add_event(test_event_set, event_code)) != PAPI_OK) {
                std::cerr << "Failed to add event " << *it << " to test event set. ";
                std::cerr << "Papi error: (" << rc << ") - " << PAPI_strerror(rc) << ". ";
                std::cerr << "This event will not be tracked.\n";
                continue;
            }

            this->_papi_event_codes.push_back(event_code);
        }

        // Clean up the test event set (we no longer need it)
        if((rc = PAPI_cleanup_eventset(test_event_set)) != PAPI_OK) {
            std::cerr << "Failed to clean up test event set. ";
            std::cerr << "Papi error: (" << rc << ") - " << PAPI_strerror(rc) << ".\n";
            exit(1);
        }

        _root = new Node(0, 0, Root);
    }

    void finalize(void)
    {
        // So far, taskwaits have been synchronized with the tasks created previously
        // But those tasks created after a given taskwait have not been connected to the taskwait
        // Note: The wd of a taskwait is always a negative number!
        // We also have to connect the root of the graph with the nodes with no parent
        // If a node with no parent cannot be connected to a taskwait, then it has to be connected to the root
        for (std::set<Node*>::const_iterator it = _graph_nodes.begin(); it != _graph_nodes.end(); ++it) {
            if (!(*it)->is_previous_synchronized()) {
                // The task has no parent, look for a taskwait|barrier suitable to be its parent
                int64_t wd_id = (*it)->get_wd_id() - (((*it)->is_concurrent() || (*it)->is_commutative()) ? concurrent_min_id : 0);
                Node* it_parent = (*it)->get_parent_task();
                Node* last_taskwait_sync = NULL;
                for (std::set<Node*>::const_iterator it2 = _graph_nodes.begin(); it2 != _graph_nodes.end(); ++it2) {
                    if ((*it)->get_wd_id()==(*it2)->get_wd_id())                    // skip current node
                        continue;
                    if ((it_parent == (*it2)->get_parent_task())                    // The two nodes are in the same region
                            && !(*it2)->is_task()                                   // The potential last sync is a taskwait|barrier|concurrent
                            && (std::abs(wd_id) > std::abs((*it2)->get_wd_id()))    // The potential last sync was created before
                            && !(*it)->is_connected_with(*it2)) {                   // Make sure we don't connect with our own child
                        if ((last_taskwait_sync == NULL)
                            || (std::abs(last_taskwait_sync->get_wd_id()) < std::abs((*it2)->get_wd_id()))) {
                                // From all suitable previous syncs. we want the one created the latest
                            last_taskwait_sync = *it2;
                        }
                    }
                }
                if (last_taskwait_sync != NULL) {
                    if (std::abs(last_taskwait_sync->get_wd_id()) < (*it)->get_wd_id())
                        Node::connect_nodes(last_taskwait_sync, *it, Synchronization, 0);
                } else {
                    Node::connect_nodes(_root, *it, Synchronization, 0);
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

        std::string base_file_name = OS::getArg(0);
        size_t slash_pos = base_file_name.find_last_of("/");

        if(slash_pos != std::string::npos) {
            base_file_name = base_file_name.substr(slash_pos+1, base_file_name.size()-slash_pos);
        }

        base_file_name += "_" + unique_key_name;

        // If pdf output was enabled, do pdf output
        if(_output_mode == "pdf") {

            // Create the dot file
            std::string const dot_file_name = print_full_graph_dot(base_file_name);
            std::string const pdf_file_name = base_file_name + ".pdf";

            // Generate the PDF from the dot file
            std::string dot_to_pdf_command = "dot -Tpdf " + dot_file_name + ".dot -o " + pdf_file_name;
            if ( system( dot_to_pdf_command.c_str( ) ) != 0 ) warning( "Could not create the pdf file." );
            std::cout << "Task Dependency Graph printed to file '" << pdf_file_name << "' in PDF format" << std::endl;
        }

        // Create the json file
        else if(_output_mode == "json") {
            print_full_graph_json(base_file_name);
        }

        else {
            std::cerr << "Task Dependency Graph not printed. Output mode '" << _output_mode << "' not recognised";
            exit(1);
        }

        // Shut down papi
        PAPI_shutdown();
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
            n->start_operation_counters(_papi_event_codes);
        }
    }

    void addSuspendTask(WorkDescriptor &w, bool last)
    {
        Node* n = find_node_from_wd_id(w.getId());
        if(n != NULL) {
            n->suspend_operation_counters(last);
            double time = (double) get_current_time() - n->get_last_time();
            n->add_total_time(time);
        }
    }

    void addEventList(unsigned int count, Event *events)
    {
        InstrumentationDictionary *iD = getInstrumentationDictionary();
        static const nanos_event_key_t create_wd_ptr = iD->getEventKey("create-wd-ptr");
        static const nanos_event_key_t dependence = iD->getEventKey("dependence");
        static const nanos_event_key_t dep_direction = iD->getEventKey("dep-direction");
        static const nanos_event_key_t dep_address = iD->getEventKey("dep-address");
        static const nanos_event_key_t dep_size = iD->getEventKey("dep-size");
        static const nanos_event_key_t user_funct_location = iD->getEventKey("user-funct-location");
        static const nanos_event_key_t taskwait = iD->getEventKey("taskwait");
        static const nanos_event_key_t critical_wd_id = iD->getEventKey("critical-wd-id");

        // Get the node corresponding to the wd_id calling this function
        // This node won't exist if the calling wd corresponds to that of the master thread
        int64_t current_wd_id = getMyWDId();
        Node* current_parent = find_node_from_wd_id(current_wd_id);

        unsigned int i;
        for(i=0; i<count; i++) {
            Event &e = events[i];
            if (e.getKey() == create_wd_ptr)
            {  // A wd is submitted => create a new node

                // Get the identifier of the task function
                WorkDescriptor *wd = (WorkDescriptor *) e.getValue();
                int64_t funct_id = (int64_t) wd->getActiveDevice().getWorkFct();

                // Get the identifier of the wd
                int64_t wd_id = wd->getId();
                _next_tw_id = std::min(_next_tw_id, -wd_id);
                _next_conc_id = wd_id + 1;
                // Create the new node
                Node* new_node = new Node(wd_id, funct_id, TaskNode);
                _graph_nodes_lock.acquire();
                _graph_nodes.insert(new_node);
                _graph_nodes_lock.release();

                // Connect the task with its parent task, if exists
                if (current_parent != NULL) {
                    Node::connect_nodes(current_parent, new_node, Nesting, 0);
                }
            }
            else if (e.getKey() == critical_wd_id)
            {
               int64_t wd_id = e.getValue();
               Node *n = find_node_from_wd_id(wd_id);
               assert(n->is_task());
               n->set_critical();
            }
            else if (e.getKey() == user_funct_location)
            {   // A user function has been called
                int64_t func_id = e.getValue();
                _funct_id_to_decl_map_lock.acquire();
                if (func_id != 0 && _funct_id_to_decl_map.find(func_id) == _funct_id_to_decl_map.end()) {
                    std::string description = iD->getValueDescription(user_funct_location, func_id);
                    if (_output_mode == "pdf") {
                        // description = func_type|func_label @ file @ line @ name_type
                        // If name_type == LABEL -> description = func_type|func_label
                        // else if name_type == FUNCTION -> description = func_type|func_label @ file @ line
                        int pos2 = description.find_first_of("@");
                        int pos3 = description.find_last_of("@")+1;
                        int pos4 = description.length();
                        std::string type = description.substr(pos3, pos4-pos3);
                        if (type == "LABEL")
                        {
                            // description = func_label                -> only store the label
                            _funct_id_to_decl_map[ func_id ] = description.substr(0, pos2);
                        }
                        else    // type == "FUNCTION"
                        {
                            // description = func_type @ file @ line   -> store the whole description
                            _funct_id_to_decl_map[ func_id ] = description.substr(0, pos3);
                        }
                    } else if(_output_mode == "json") {
                        _funct_id_to_decl_map[ func_id ] = description;
                    } else {
                        _funct_id_to_decl_map[ func_id ] = description;
                    }
                }
                _funct_id_to_decl_map_lock.release();
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
                    case 8:     dep_type = OutAny;
                                receiver_wd_id += concurrent_min_id;    // wd -> common
                                break;
                    case 9:     dep_type = InAny;
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
                        _graph_nodes_lock.acquire();
                        _graph_nodes.insert(sender);
                        _graph_nodes_lock.release();
                    }

                    // Create the new receiver node, if necessary
                    if(receiver == NULL) {
                        receiver = new Node(receiver_wd_id, 0, nt);
                        _graph_nodes_lock.acquire();
                        _graph_nodes.insert(receiver);
                        _graph_nodes_lock.release();
                    }
                }

                // Consume dep_address event key that prcedes data_size
                e = events[++i];
                assert(e.getKey() == dep_address);

                // Get dependency data size
                e = events[++i];
                assert(e.getKey() == dep_size);
                uint64_t dep_data_size = e.getValue();

                Node::connect_nodes(sender, receiver, Dependency, dep_data_size, dep_type);
            }
            else if (e.getKey() == taskwait)
            {   // A taskwait occurs
                // Synchronize all previous nodes created by the same task that have not been yet synchronized
                Node* new_node = new Node(_next_tw_id, -1, TaskwaitNode);
                --_next_tw_id;
                // First synchronize the tasks
                _graph_nodes_lock.acquire();
                for(std::set<Node*>::const_iterator it = _graph_nodes.begin(); it != _graph_nodes.end(); ++it)
                {
                    // Synchronization nodes will be connected, if necessary, when 'finalize' is called
                    if (!(*it)->is_next_synchronized()
                            && ((*it)->is_task() || (*it)->is_concurrent() || (*it)->is_commutative())) {
                        Node* parent_task = (*it)->get_parent_task();
                        if(current_parent == parent_task) {
                            Node::connect_nodes(*it, new_node, Synchronization, 0);
                        }
                    }
                }
                _graph_nodes.insert(new_node);
                _graph_nodes_lock.release();
            }
        }
    }

    void threadStart(BaseThread &thread) {}

    void threadFinish (BaseThread &thread) {}

#endif

};

std::list<std::string> InstrumentationTGDump::_papi_event_args;
std::string InstrumentationTGDump::_output_mode;

namespace ext {

    class InstrumentationTGDumpPlugin : public Plugin {
    public:
        InstrumentationTGDumpPlugin () : Plugin("Instrumentation plugin which prints the graph to a dot file.", 1) {}
        ~InstrumentationTGDumpPlugin () {}

        void config(Config &cfg)
        {
            cfg.setOptionsSection(
                "Task graph dump plugin ", "tg-dump specific options");

            InstrumentationTGDump::_papi_event_args.push_back("PAPI_FP_OPS");
            InstrumentationTGDump::_papi_event_args.push_back("PAPI_TOT_INS");
            InstrumentationTGDump::_papi_event_args.push_back("PAPI_TOT_CYC");

            cfg.registerConfigOption(
                "papi-events", NEW Config::StringVarList(InstrumentationTGDump::_papi_event_args),
                "Defines which PAPI events to track for tasks.");
            cfg.registerArgOption(
                "papi-events", "papi-events");

            InstrumentationTGDump::_output_mode = "json";

            cfg.registerConfigOption(
                "tgdump-mode", NEW Config::StringVar(InstrumentationTGDump::_output_mode),
                "Specifies how tg-dump should output the task graph. "
                "Accepted options are: json (default), pdf");
            cfg.registerArgOption(
                "tgdump-mode", "tgdump-mode");
        }

        void init ()
        {
            if(InstrumentationTGDump::_output_mode != "json" && InstrumentationTGDump::_output_mode != "pdf") {
                std::cerr << "Invalid tgdump-mode '" << InstrumentationTGDump::_output_mode << "'. ";
                std::cerr << "One of the following expected: 'json', 'pdf'. ";
                std::cerr << "Using default 'json'." << std::endl;
                InstrumentationTGDump::_output_mode = "json";
            }

            sys.setInstrumentation(new InstrumentationTGDump());
        }
    };

} // namespace ext

} // namespace nanos

DECLARE_PLUGIN("instrumentation-tg-dump",nanos::ext::InstrumentationTGDumpPlugin);
