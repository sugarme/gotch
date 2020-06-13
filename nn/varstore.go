package nn

import (
	// "fmt"
	"fmt"
	"sync"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

// SEP is a separator to separate path elements in the tensor names.
const SEP = "."

// Variables represents a collection of tensors.
//
// NOTE: When the variable store is frozen, trainable still is set to tree,
// however the tensor is not set to require gradients.
type Variables struct {
	mutex             *sync.Mutex
	NamedVariables    map[string]ts.Tensor
	TrainableVariable []ts.Tensor
}

// VarStore is used to store variables used by one or multiple layers.
// It specifies a SINGLE device where all variables are stored.
type VarStore struct {
	device    gotch.Device
	variables Variables // TODO: should we export this field
}

// Path is variable store with an associated path for variables naming.
type Path struct {
	path     []string
	varstore VarStore
}

// Entry holds an entry corresponding to a given name in Path.
type Entry struct {
	name      string
	variables Variables // MutexGuard
	path      Path
}

// NewVarStore creates a new variable store located on the specified device
func NewVarStore(device gotch.Device) VarStore {
	variables := Variables{
		mutex:             &sync.Mutex{},
		NamedVariables:    make(map[string]ts.Tensor, 0),
		TrainableVariable: make([]ts.Tensor, 0),
	}

	return VarStore{
		device:    device,
		variables: variables,
	}
}

// Device returns device for this var-store
func (vs *VarStore) Device() gotch.Device {
	return vs.device
}

// Len returns the number of tensors currently stored on this var-store
func (vs *VarStore) Len() (retVal int) {
	vs.variables.mutex.Lock()
	retVal = len(vs.variables.NamedVariables)
	vs.variables.mutex.Unlock()

	return retVal
}

// IsEmpty returns true if no tensors are currently stored on this var-store
func (vs *VarStore) IsEmpty() (retVal bool) {
	vs.variables.mutex.Lock()
	retVal = (len(vs.variables.NamedVariables) == 0)
	vs.variables.mutex.Unlock()

	return retVal
}

// TrainableVariables returns all trainable variables for this var-store
func (vs *VarStore) TrainableVariable() (retVal []ts.Tensor) {
	vs.variables.mutex.Lock()
	retVal = vs.variables.TrainableVariable
	vs.variables.mutex.Unlock()

	return retVal
}

// Variables returns all variables and their names in a map[variable_name]Tensor
func (vs *VarStore) Variables() (retVal map[string]ts.Tensor) {
	vs.variables.mutex.Lock()
	retVal = vs.variables.NamedVariables
	vs.variables.mutex.Unlock()

	return retVal
}

// Root gets the root path for this var-store
//
// NOTE: Variables are named and organized using paths. This function returns
// the top level path for the var store and can be combined with '/'
// to create sub-paths.
func (vs *VarStore) Root() (retVal Path) {
	return Path{
		path:     []string{},
		varstore: *vs,
	}
}

// Save saves the var-store variable values to a file
//
// NOTE: Weight values for all the tensors currently stored in the
// var-store gets saved in the given file.
func (vs *VarStore) Save(filepath string) (err error) {
	vs.variables.mutex.Lock()
	namedTensorsMap := vs.variables.NamedVariables
	vs.variables.mutex.Unlock()

	// Convert map to []NamedTensor
	var namedTensors []ts.NamedTensor
	for k, v := range namedTensorsMap {
		namedTensors = append(namedTensors, ts.NamedTensor{
			Name:   k,
			Tensor: v,
		})
	}

	return ts.SaveMulti(namedTensors, filepath)
}

// Load loads the var-store variable values from a file.
//
// NOTE: Weight values for all the tensors currently stored in the
// var-store gets loaded from the given file. Note that the set of
// variables stored in the var-store is not changed, only the values
// for these tensors are modified.
// It will throw error if name of the loaded tensors can not find
// in the current var-store named tensors set.
func (vs *VarStore) Load(filepath string) (err error) {
	namedTensors, err := ts.LoadMultiWithDevice(filepath, vs.device)
	if err != nil {
		return err
	}

	var currMap map[string]ts.Tensor

	// Match and in-place copy value (update) from newly loaded tensors
	// to existing named tensors if name is matched. Throw error otherwise.
	vs.variables.mutex.Lock()
	currMap = vs.variables.NamedVariables
	vs.variables.mutex.Unlock()

	for _, namedTs := range namedTensors {
		if currTs, ok := currMap[namedTs.Name]; !ok {
			err = fmt.Errorf("Cannot find tensor with name: %v in variable store. \n", namedTs.Name)
			return err
		}

		// It's matched. Now, copy in-place the loaded tensor value to var-store
		// TODO: implement it
		// 1. Copy in-place with `f_copy_` for `currTs`

		// 2. Call `NoGrad` on newly updated tensor value
		// noGradTs, err := ts.NoGrad(namedTs.Tensor)
		// if err != nil {
		// return err
		// }
	}

	return nil
}
