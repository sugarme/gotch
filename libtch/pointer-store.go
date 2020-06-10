package libtch

// #include <stdlib.h>
import "C"

import (
	"sync"
	"unsafe"
)

// PointerStore is a Go struct to deal with the Go use case that
// we can not pass Go pointer to C. In other words, it is used to solve
// error: "panic: runtime error: cgo argument has Go pointer to Go pointer"
//
// NOTE: the concept is taken from: https://github.com/mattn/go-pointer
//
// Example:
//     store := NewPointerStore()
//     type Car struct{Name string, Model string}
//     var landy Car{Name: "Defender", Model: "99"}
//     landyPtr := store.Set(landy)
//     landy = store.Get(landyPtr).(Car)
//     store.Free(landyPtr)
type PointerStore struct {
	store map[unsafe.Pointer]interface{}
}

// NewPointerStore creates a new PointerStore
func NewPointerStore() PointerStore {
	store := map[unsafe.Pointer]interface{}{}
	return PointerStore{
		store: store,
	}
}

// create a locker for pointer store
var mutex sync.Mutex

// Set stores value to pointer store and returns a unsafe.Pointer
//
// NOTE: This is a little hacky. As Go doesn't allow C code to store Go pointers,
// a one-byte C pointer is created for indexing purpose.
//
// Example:
// TODO: an example
func (ps PointerStore) Set(v interface{}) unsafe.Pointer {
	if v == nil {
		return nil
	}

	var ptr unsafe.Pointer = C.malloc(C.size_t(1))
	if ptr == nil {
		panic("Cannot create C pointer for 'PointerStore'.")
	}

	mutex.Lock()
	ps.store[ptr] = v
	mutex.Unlock()

	return ptr
}

// Get get value back from pointer store
//
// Example:
// TODO: an example
func (ps PointerStore) Get(ptr unsafe.Pointer) (v interface{}) {
	if ptr == nil {
		return nil
	}

	mutex.Lock()
	v = ps.store[ptr]
	mutex.Unlock()
	return
}

// Delete removes pointer from pointer store and frees up memory.
//
// Example:
// TODO: an example
func (ps PointerStore) Free(ptr unsafe.Pointer) {
	if ptr == nil {
		return
	}

	mutex.Lock()
	delete(ps.store, ptr)
	mutex.Unlock()

	C.free(ptr)
}
