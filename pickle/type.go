package pickle

import (
	"container/list"
	"fmt"
	"reflect"
)

// This file contains custom types and interfaces for casting Python data types from Pickle.
//
// Custom Types:
// =============
// 1. ByteArray
// 2. Dict
// 3. Tuple
// 4. OrderedDict
// 5. List
// 6. Set
// 7. FrozenSet
// 8. Object
// 9. Reconstructor
// 10. GenericClass

// Interfaces:
// ===========

// Callable is implemented by any value that can be directly called to get a
// new value.
//
// It is usually implemented by Python-like functions (returning a value
// given some arguments), or classes (typically returning an instance given
// some constructor arguments).
type Callable interface {
	// Call mimics a direct invocation on a Python value, such as a function
	// or class (constructor).
	Call(args ...interface{}) (interface{}, error)
}

// PyNewable is implemented by any value that has a Python-like
// "__new__" method.
//
// It is usually implemented by values representing Python classes.
type PyNewable interface {
	// PyNew mimics Python invocation of the "__new__" method, usually
	// provided by classes.
	//
	// See: https://docs.python.org/3/reference/datamodel.html#object.__new__
	PyNew(args ...interface{}) (interface{}, error)
}

// PyStateSettable is implemented by any value that has a Python-like
// "__setstate__" method.
type PyStateSettable interface {
	// PySetState mimics Python invocation of the "__setstate__" method.
	//
	// See: https://docs.python.org/3/library/pickle.html#object.__setstate__
	PySetState(state interface{}) error
}

// PyDictSettable is implemented by any value that can store dictionary-like
// key/value pairs. It reflects Python behavior of setting a key/value pair on
// an object's "__dict__" attribute.
type PyDictSettable interface {
	// PyDictSet mimics the setting of a key/value pair on an object's
	//"__dict__" attribute.
	//
	// See: https://docs.python.org/3/library/stdtypes.html#object.__dict__
	PyDictSet(key, value interface{}) error
}

// PyAttrSettable is implemented by any value on which an existing or new
// Python-like attribute can be set. In Python this is done with "setattr"
// builtin function.
type PyAttrSettable interface {
	// PySetAttr mimics the setting of an arbitrary value to an object's
	// attribute.
	//
	// In Python this is done with "setattr" function, to which object,
	// attribute name, and value are passed. For an easy and clear
	// implementation, here instead we require this method to be implemented
	// on the "object" itself.
	//
	// See: https://docs.python.org/3/library/functions.html#setattr
	PySetAttr(key string, value interface{}) error
}

// ByteArray:
//===========

// ByteArray simulates Python bytearray.
type ByteArray []byte

func NewByteArray() *ByteArray {
	arr := make(ByteArray, 0)
	return &arr
}

func NewByteArrayFromSlice(slice []byte) *ByteArray {
	arr := ByteArray(slice)
	return &arr
}

func (a *ByteArray) Get(i int) byte {
	return (*a)[i]
}

func (a *ByteArray) Len() int {
	return len(*a)
}

// Dict:
//======

// DictSetter is implemented by any value that exhibits a dict-like behaviour,
// allowing arbitrary key/value pairs to be set.
type DictSetter interface {
	Set(key, value interface{})
}

// Dict represents a Python "dict" (builtin type).
//
// It is implemented as a slice, instead of a map, because in Go not
// all types can be map's keys (e.g. slices).
type Dict []*DictEntry

type DictEntry struct {
	Key   interface{}
	Value interface{}
}

// NewDict makes and returns a new empty Dict.
func NewDict() *Dict {
	d := make(Dict, 0)
	return &d
}

// Set sets into the Dict the given key/value pair.
func (d *Dict) Set(key, value interface{}) {
	*d = append(*d, &DictEntry{
		Key:   key,
		Value: value,
	})
}

// Get returns the value associated with the given key (if any), and whether
// the key is present or not.
func (d *Dict) Get(key interface{}) (interface{}, bool) {
	for _, entry := range *d {
		if reflect.DeepEqual(entry.Key, key) {
			return entry.Value, true
		}
	}
	return nil, false
}

// MustGet returns the value associated with the given key, if if it exists,
// otherwise it panics.
func (d *Dict) MustGet(key interface{}) interface{} {
	value, ok := d.Get(key)
	if !ok {
		panic(fmt.Errorf("key not found in Dict: %#v", key))
	}
	return value
}

// Len returns the length of the Dict, that is, the amount of key/value pairs
// contained by the Dict.
func (d *Dict) Len() int {
	return len(*d)
}

var _ DictSetter = &Dict{}

// Tuple:
// ======

type Tuple []interface{}

func NewTupleFromSlice(slice []interface{}) *Tuple {
	t := Tuple(slice)
	return &t
}

func (t *Tuple) Get(i int) interface{} {
	return (*t)[i]
}

func (t *Tuple) Len() int {
	return len(*t)
}

// OrderedDict:
// ============

// OrderedDictClass represent Python "collections.OrderedDict" class.
//
// This class allows the indirect creation of OrderedDict objects.
type OrderedDictClass struct{}

var _ Callable = &OrderedDictClass{}

// Call returns a new empty OrderedDict. It is equivalent to Python
// constructor "collections.OrderedDict()".
//
// No arguments are supported.
func (*OrderedDictClass) Call(args ...interface{}) (interface{}, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf(
			"OrderedDictClass.Call args not supported: %#v", args)
	}
	return NewOrderedDict(), nil
}

// OrderedDict is a minimal and trivial implementation of an ordered map,
// which represent a Python "collections.OrderedDict" object.
//
// It is composed by a simple unordered Map, and a List to keep the order of
// the entries. The former is useful for direct key lookups, the latter for
// iteration.
type OrderedDict struct {
	// Map associates a key of any type (interface{}) to OrderedDictEntry
	// pointer values. These values are shared with List.
	Map map[interface{}]*OrderedDictEntry
	// List is an ordered list of OrderedDictEntry pointers, which are
	// also shared with Map.
	List *list.List
	// PyDict represents Python "object.__dict__" dictionary of attributes.
	PyDict map[string]interface{}
}

var _ DictSetter = &OrderedDict{}
var _ PyDictSettable = &OrderedDict{}

// OrderedDictEntry is a single key/value pair stored in an OrderedDict.
//
// A pointer to an OrderedDictEntry is always shared between OrderedDict's Map
// and List.
type OrderedDictEntry struct {
	// Key of a single OrderedDict's entry.
	Key interface{}
	// Value of a single OrderedDict's entry.
	Value interface{}
	// ListElement is a pointer to the OrderedDict's List Element which
	// contains this very OrderedDictEntry.
	ListElement *list.Element
}

// NewOrderedDict makes and returns a new empty OrderedDict.
func NewOrderedDict() *OrderedDict {
	return &OrderedDict{
		Map:    make(map[interface{}]*OrderedDictEntry),
		List:   list.New(),
		PyDict: make(map[string]interface{}),
	}
}

// Set sets into the OrderedDict the given key/value pair. If the key does not
// exist yet, the new pair is positioned at the end (back) of the OrderedDict.
// If the key already exists, the existing associated value is replaced with the
// new one, and the original position is maintained.
func (o *OrderedDict) Set(k, v interface{}) {
	if entry, ok := o.Map[k]; ok {
		entry.Value = v
		return
	}

	entry := &OrderedDictEntry{
		Key:   k,
		Value: v,
	}
	entry.ListElement = o.List.PushBack(entry)
	o.Map[k] = entry
}

// Get returns the value associated with the given key (if any), and whether
// the key is present or not.
func (o *OrderedDict) Get(k interface{}) (interface{}, bool) {
	entry, ok := o.Map[k]
	if !ok {
		return nil, false
	}
	return entry.Value, true
}

// MustGet returns the value associated with the given key, if if it exists,
// otherwise it panics.
func (o *OrderedDict) MustGet(key interface{}) interface{} {
	value, ok := o.Get(key)
	if !ok {
		panic(fmt.Errorf("key not found in OrderedDict: %#v", key))
	}
	return value
}

// Len returns the length of the OrderedDict, that is, the amount of key/value
// pairs contained by the OrderedDict.
func (o *OrderedDict) Len() int {
	return len(o.Map)
}

// PyDictSet mimics the setting of a key/value pair on Python "__dict__"
// attribute of the OrderedDict.
func (o *OrderedDict) PyDictSet(key, value interface{}) error {
	sKey, keyOk := key.(string)
	if !keyOk {
		return fmt.Errorf(
			"OrderedDict.PyDictSet() requires string key: %#v", key)
	}
	o.PyDict[sKey] = value
	return nil
}

// List:
// =====

// ListAppender is implemented by any value that exhibits a list-like
// behaviour, allowing arbitrary values to be appended.
type ListAppender interface {
	Append(v interface{})
}

// List represents a Python "list" (builtin type).
type List []interface{}

var _ ListAppender = &List{}

// NewList makes and returns a new empty List.
func NewList() *List {
	l := make(List, 0)
	return &l
}

// NewListFromSlice makes and returns a new List initialized with the elements
// of the given slice.
//
// The new List is a simple type cast of the input slice; the slice is _not_
// copied.
func NewListFromSlice(slice []interface{}) *List {
	l := List(slice)
	return &l
}

// Append appends one element to the end of the List.
func (l *List) Append(v interface{}) {
	*l = append(*l, v)
}

// Get returns the element of the List at the given index.
//
// It panics if the index is out of range.
func (l *List) Get(i int) interface{} {
	return (*l)[i]
}

// Len returns the length of the List.
func (l *List) Len() int {
	return len(*l)
}

// Set:
// ====

// SetAdder is implemented by any value that exhibits a set-like behaviour,
// allowing arbitrary values to be added.
type SetAdder interface {
	Add(v interface{})
}

// Set represents a Python "set" (builtin type).
//
// It is implemented in Go as a map with empty struct values; the actual set
// of generic "interface{}" items is thus represented by all the keys.
type Set map[interface{}]setEmptyStruct

var _ SetAdder = &Set{}

type setEmptyStruct struct{}

// NewSet makes and returns a new empty Set.
func NewSet() *Set {
	s := make(Set)
	return &s
}

// NewSetFromSlice makes and returns a new Set initialized with the elements
// of the given slice.
func NewSetFromSlice(slice []interface{}) *Set {
	s := make(Set, len(slice))
	for _, item := range slice {
		s[item] = setEmptyStruct{}
	}
	return &s
}

// Len returns the length of the Set.
func (s *Set) Len() int {
	return len(*s)
}

// Add adds one element to the Set.
func (s *Set) Add(v interface{}) {
	(*s)[v] = setEmptyStruct{}
}

// Has returns whether the given value is present in the Set (true)
// or not (false).
func (s *Set) Has(v interface{}) bool {
	_, ok := (*s)[v]
	return ok
}

// FrozenSet:
//===========

// FrozenSet represents a Python "frozenset" (builtin type).
//
// It is implemented in Go as a map with empty struct values; the actual set
// of generic "interface{}" items is thus represented by all the keys.
type FrozenSet map[interface{}]frozenSetEmptyStruct

type frozenSetEmptyStruct struct{}

// NewFrozenSetFromSlice makes and returns a new FrozenSet initialized
// with the elements of the given slice.
func NewFrozenSetFromSlice(slice []interface{}) *FrozenSet {
	f := make(FrozenSet, len(slice))
	for _, item := range slice {
		f[item] = frozenSetEmptyStruct{}
	}
	return &f
}

// Len returns the length of the FrozenSet.
func (f *FrozenSet) Len() int {
	return len(*f)
}

// Has returns whether the given value is present in the FrozenSet (true)
// or not (false).
func (f *FrozenSet) Has(v interface{}) bool {
	_, ok := (*f)[v]
	return ok
}

// Object:
//========

type ObjectClass struct{}

var _ PyNewable = &ObjectClass{}

func (o *ObjectClass) PyNew(args ...interface{}) (interface{}, error) {
	if len(args) == 0 {
		return nil, fmt.Errorf("ObjectClass.PyNew called with no arguments")
	}
	switch class := args[0].(type) {
	case PyNewable:
		return class.PyNew()
	default:
		return nil, fmt.Errorf(
			"ObjectClass.PyNew unprocessable args: %#v", args)
	}
}

// Reconstructor:
//===============

type Reconstructor struct{}

var _ Callable = &Reconstructor{}

func (r *Reconstructor) Call(args ...interface{}) (interface{}, error) {
	if len(args) < 2 {
		return nil, fmt.Errorf("Reconstructor: invalid arguments: %#v", args)
	}
	class := args[0]
	switch base := args[1].(type) {
	case PyNewable:
		return base.PyNew(class)
	default:
		return nil, fmt.Errorf(
			"Reconstructor: unprocessable arguments: %#v", args)
	}
}

// GenericClass:
//==============

type GenericClass struct {
	Module string
	Name   string
}

var _ PyNewable = &GenericClass{}

type GenericObject struct {
	Class           *GenericClass
	ConstructorArgs []interface{}
}

func NewGenericClass(module, name string) *GenericClass {
	return &GenericClass{Module: module, Name: name}
}

func (g *GenericClass) PyNew(args ...interface{}) (interface{}, error) {
	return &GenericObject{
		Class:           g,
		ConstructorArgs: args,
	}, nil
}

// getThnnFunctionBackend is for historical pickle deserilaization, it is not used otherwise
type getThnnFunctionBackend struct{}

var _ Callable = &getThnnFunctionBackend{}

func (getThnnFunctionBackend) Call(_ ...interface{}) (interface{}, error) {
	return nil, nil
}
