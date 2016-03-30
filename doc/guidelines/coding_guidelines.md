**Back to:** [README](../../README.md) > [developers guide](../developers_guide.md) >

Coding Guidelines
=================

* Indenting should use 3 spaces (not tabs!).
* Brackets should start on the opening statement (if, for,...) except for function blocks.
* Labels, classes, cases should all be indented.
* Spaces should be used inside parenthesis (e.g., ( 5 ) not (5) )
* If in doubt you can use astyle (http://astyle.sourceforge.net). Check script/astyle-nanox.sh script.
* All files (sources, headers, Makefile.am) MUST have the proper license header (see script/preamble.txt). You can use headache to help you.
* Type names should (in general) be defined inside the scope that uses them.
* Error problems should be reported through the exception mechanism.
* Do not declare static members in the core library outside of the System class unless you know very well what you are doing (this can mess the initialization process).
* Use STL containers an STL algorithms when possible. Beware though, of thread-safety implications with STL.
* If an argument should never be NULL declare it as a reference (when possible).
* Remember to define default values for arguments where it makes sense (i.e. most of the time is that value).
* Avoid the usage of macros unless strictly necessary. Consider templates and inline functions first.
* Use static functions when possible (particularly inside plugins).
* Enum types should in general be defined inside the class that uses them.

Naming Conventions
------------------
* All core library classes must be defined in the nanos namespace.
* All plugin classes must be defined in the nanos::ext namespace:
<pre>
   namespace nanos {
   namespace ext {
     class PluginClass
     {
        ...
     };
   }}</pre>
* Class names should have all words capitalized with no _  (e.g. !MyClassName)
* Class methods should have all words capitalized with no _ except for the first one (e.g. myMethod)
* Class data members should start with _ and have all words capitalized except the first one (e.g. _myData)
* Other variables (globals, locals, parameters,...)  should have no words capitalized using _ to separate them (e.g. my_data)
* Type names should have no words capitalized using _ to separate them and ending with _t (e.g. my_type_t)
* Define typedefs for template containers used for data members (e.g. typedef std::vector<T*> my_type_t;)
* Define typedefs for enum types used for data members (e.g. typedef enum{X,Y,Z} my_type_t;)
* External symbols (functions and type names) MUST be prefixed with nanos_ (e.g. nanos_wd_t)

Classes and Structs
-------------------
* Constructors, destructors and assignment operators
  * All classes MUST have an explicit default constructor and copy constructor.
  * Remember to initialize all data members in the constructors. When possible do so in the initialization list and not in the constructor body.
  * All classes MUST have an explicit destructor unless they have no data members.
  * All classes MUST have an explicit assignment operator.
  * If you implement an assignment operator always check if a self-assignment ( a = a; ) is handled correctly. Mark so in the code with a comment.
  * If you think the class should not have any of them declare as private
* As a rule of thumb, all data members should be private with get/set methods defined in the headers to increase inlining. Consider it deeply before declaring a data member shared or friend to another class.
* All classes that implement a virtual method MUST declare their destructor to be virtual.
* Do NOT call virtual methods from inside constructors or destructors (see http://www.artima.com/cppsource/nevercall.html)
* Use the named parameter idiom when possible to set object properties (see WorkDescriptor::tied for an example).

Methods and Functions
---------------------
* Function declarations (or definitions) MUST have an space between function name and the opening parentheses:
<pre>
   type_t myMethod (type_foo_t foo, ..., type_bar_t bar);
</pre>
* Function calls have no spaces between function name and the opening parentheses:
<pre>
   myMethod(foo, ..., bar);
</pre>
* Parameters (or arguments) in multiple lines should be indented to be below the previous line parameters:
<pre>
   type_t myMethod (type_foo_t foo, ..., type_bar_t bar,
                    type_baz_t baz);
</pre>
* Use of default parameters is preferred over overloading a method or function (when possible).

Conditional
-----------
* Compare pointers to NULL:
<pre>
   if ( p != NULL ) {
      ...
   }
</pre>
* and not:
<pre>
   if (!p) {
      ...
   }
</pre>
* Compare integer to the appropriate value:
<pre>
   if ( i > 0 ) {
      ...
   }
</pre>
* and not:
<pre>
   if (i) {
      ...
   }
</pre>
* Do not compare booleans to true/false:
<pre>
   if ( myBoolean ) {
      ...
   }
</pre>
* and not:
<pre>
   if ( myBoolean == true ) {
      ...
   }
</pre>
* Else [else-if] statement will close and open brackets in the same line:
<pre>
   if ( cond ) {
      ...
   } else {
      ... 
   }
</pre>

Other Programming Patterns
--------------------------

* Avoid if possible any kind of atomic operation (atomic updates, locks, cas, ...). If necessary use the double-checked locking optimization:
<pre>
   if ( cond ) {
      lock++;
      if ( cond ) {
         // do whatever
      }
      lock--;
   }
</pre>

Code Documentation
------------------

Documenting Classes and Data Members
------------------------------------

Documenting Functions and Methods
---------------------------------

* Use doxygen to document all function and method definitions.
  * A brief summary (one sentence) declaring the function main purpose (mandatory).
  * A list of parameters (with its type [in], [out] or [in,out]) including a short description (mandatory, if any).
  * The returning value of the function (mandatory, if non-void).
  * A more detailed function description (when needed).
  * Use the ''See Also'' section to refer other documented functions, types, structures... (when needed)
* A doxygen example:
<pre>
//! \brief Brief function description
//! \param [type] param1 Param 1 description
//! \param [type] param2 Param 2 description
//! ...
//! \param [type] paramN Param N description
//! \return Return value description
//! \par Description:
//! You can use several lines to describe the function. Doxygen will text-justify the provided
//! description in order to fit it properly in the web page.
//!
//! \sa SeeAlsoItem1, SeeAlsoItem2, ... SeeAlsoItemN
void nanosFunction ( type1_t param1, type2_t param2, ... ,typeN_t paramN )
{
   ...
}
</pre>
* Use doxygen to document as well data members, enum values and global variables.
<pre>
type1_t _data1 //!< data1 description
type2_t _data2 //!< data2 description
...
typeN_t _dataN //!< dataN description
</pre>
* Document parts of your code where what is happening may not be obvious. 
<pre>
	// copy all q elements into p, q is zero-terminated
	while ( *p++ = *q++ ); 
</pre>
* Use FIXME (#issue) and TODO (#issue) to mark as 'defect' or 'task' respectively.
<pre>
   // FIXME: (#104) Memory is requiered to be aligned to 8 bytes in some architectures, temporary solved using: ((s+7)>>3)<<3
   int size_to_allocate = ( ( *uwd == NULL ) ? sizeof( WD ) : 0 ) + ( ( data != NULL && *data == NULL ) ? (((data_size+7)>>3)<<3) : 0 ) +
                          sizeof( DD* ) * num_devices + dd_size ; 
</pre>

Commandline and environment variables
-------------------------------------

* All parameters names should start with -nth (e.g. -nth-pes ) and they should be lowercase
* Use '-' in parameter names to separate words
* If it's not a core parameter it should also include a plugin prefix (e.g., -nth-smp-stack-size, -nth-wfsch-fifo )
* All environment variable should start with NTH and they should be uppercase
* Use '_' in enviroment variables to separate words

