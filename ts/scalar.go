package ts

import (
	"fmt"
	"log"
	"runtime"
	"sync/atomic"

	"github.com/sugarme/gotch"
	lib "github.com/sugarme/gotch/libtch"
)

type Scalar struct {
	name    string
	cscalar lib.Cscalar
}

// free releases C allocated memory.
func freeCScalar(x *Scalar) error {
	if gotch.Debug {
		nbytes := x.nbytes()
		atomic.AddInt64(&AllocatedMem, -nbytes)

		log.Printf("INFO: Released scalar %q - C memory: %d bytes.\n", x.name, nbytes)
	}
	lock.Lock()
	delete(ExistingScalars, x.name)
	lock.Unlock()

	lib.AtsFree(x.cscalar)
	if err := TorchErr(); err != nil {
		return err
	}

	return nil
}

func newScalarName(nameOpt ...string) string {
	var name string
	if len(nameOpt) > 0 {
		name = nameOpt[0]
	} else {
		name = fmt.Sprintf("tensor_%09d", TensorCount)
	}

	return name
}

func newScalar(cscalar lib.Cscalar, nameOpt ...string) *Scalar {
	x := &Scalar{
		cscalar: cscalar,
	}

	atomic.AddInt64(&ScalarCount, 1)
	if gotch.Debug {
		nbytes := x.nbytes()
		atomic.AddInt64(&AllocatedMem, nbytes)

		log.Printf("INFO: scalar %q added - Allocated memory (%d bytes).\n", x.name, nbytes)
	}
	lock.Lock()
	x.name = newName(nameOpt...)
	ExistingScalars[x.name] = struct{}{}
	lock.Unlock()

	runtime.SetFinalizer(x, freeCScalar)

	return x
}

func (sc *Scalar) nbytes() int64 {
	return 4 // either Int64 or Float64 scalar -> 4 bytes
}

// IntScalar creates a integer scalar
func IntScalar(v int64) *Scalar {
	cscalar := lib.AtsInt(v)
	return newScalar(cscalar)
}

// FloatScalar creates a float scalar
func FloatScalar(v float64) *Scalar {
	cscalar := lib.AtsFloat(v)
	return newScalar(cscalar)
}

// ToInt returns a integer value
func (sc *Scalar) ToInt() (retVal int64, err error) {
	retVal = lib.AtsToInt(sc.cscalar)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	return retVal, nil
}

// ToFloat returns a float value
func (sc *Scalar) ToFloat() (retVal float64, err error) {
	retVal = lib.AtsToFloat(sc.cscalar)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	return retVal, nil
}

// ToString returns a string representation of scalar value
func (sc *Scalar) ToString() (retVal string, err error) {
	retVal = lib.AtsToString(sc.cscalar)
	err = TorchErr()
	if err != nil {
		return retVal, err
	}

	return retVal, nil
}

// Drop sets scalar to zero and frees up C memory
//
// TODO: Really? after running s.Drop() and s.ToInt()
// it returns Zero.
func (sc *Scalar) Drop() (err error) {
	// TODO. FIXME either remove or rewind for specific scenario
	return nil
	// lib.AtsFree(sc.cscalar)
	// return TorchErr()
}

func (sc *Scalar) MustDrop() {
	// TODO. FIXME either remove or rewind for specific scenario
	return
	// lib.AtsFree(sc.cscalar)
	// if err := TorchErr(); err != nil {
	// log.Fatal(err)
	// }
}
