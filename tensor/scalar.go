package tensor

import (
	// "unsafe"
	"log"

	lib "github.com/sugarme/gotch/libtch"
)

type Scalar struct {
	cscalar lib.Cscalar
}

// IntScalar creates a integer scalar
func IntScalar(v int64) *Scalar {
	cscalar := lib.AtsInt(v)
	return &Scalar{cscalar}
}

// FloatScalar creates a float scalar
func FloatScalar(v float64) *Scalar {
	cscalar := lib.AtsFloat(v)
	return &Scalar{cscalar}
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
	lib.AtsFree(sc.cscalar)
	return TorchErr()
}

func (sc *Scalar) MustDrop() {
	lib.AtsFree(sc.cscalar)
	if err := TorchErr(); err != nil {
		log.Fatal(err)
	}

}
