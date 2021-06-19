package nn

import (
	"fmt"
	"log"
	"reflect"
	"strings"
	"sync"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

// SEP is a separator to separate path elements in the tensor names.
const SEP = "."

type Var struct {
	Tensor *ts.Tensor
	Group  uint // optimizer parameter group
}

// Variables represents a collection of tensors.
//
// NOTE: When the variable store is frozen, trainable still is set to tree,
// however the tensor is not set to require gradients.
type Variables struct {
	mutex              *sync.Mutex
	NamedVariables     map[string]*ts.Tensor
	TrainableVariables []Var
	// TrainableVariables []ts.Tensor
}

// VarStore is used to store variables used by one or multiple layers.
// It specifies a SINGLE device where all variables are stored.
type VarStore struct {
	device gotch.Device
	Vars   Variables
}

// Path is variable store with an associated path for variables naming.
type Path struct {
	path     []string
	varstore *VarStore
	group    uint // optimizer parameter group
}

// Entry holds an entry corresponding to a given name in Path.
type Entry struct {
	name      string
	variables *Variables // MutexGuard
	path      *Path
}

// NewVarStore creates a new variable store located on the specified device
func NewVarStore(device gotch.Device) *VarStore {
	variables := Variables{
		mutex:              &sync.Mutex{},
		NamedVariables:     make(map[string]*ts.Tensor, 0),
		TrainableVariables: make([]Var, 0),
		// TrainableVariables: make([]ts.Tensor, 0),
	}

	return &VarStore{
		device: device,
		Vars:   variables,
	}
}

// NOTE:
// To get (initiate) a path, call vs.Root()

// VarStore methods:
// =================

// Device returns device for this var-store
func (vs *VarStore) Device() gotch.Device {
	return vs.device
}

// Len returns the number of tensors currently stored on this var-store
func (vs *VarStore) Len() int {
	vs.Vars.mutex.Lock()
	defer vs.Vars.mutex.Unlock()
	return len(vs.Vars.NamedVariables)
}

// IsEmpty returns true if no tensors are currently stored on this var-store
func (vs *VarStore) IsEmpty() bool {
	vs.Vars.mutex.Lock()
	defer vs.Vars.mutex.Unlock()
	return (len(vs.Vars.NamedVariables) == 0)
}

// TrainableVariabless returns all trainable variables for this var-store
func (vs *VarStore) TrainableVariables() []ts.Tensor {
	vs.Vars.mutex.Lock()
	defer vs.Vars.mutex.Unlock()

	var retVal []ts.Tensor
	for _, v := range vs.Vars.TrainableVariables {
		retVal = append(retVal, *v.Tensor.MustShallowClone())
	}

	return retVal
}

// Variables returns all variables and their names in a map[variable_name]Tensor
func (vs *VarStore) Variables() map[string]*ts.Tensor {
	vs.Vars.mutex.Lock()
	defer vs.Vars.mutex.Unlock()

	namedTensors := make(map[string]*ts.Tensor, 0)

	for k, v := range vs.Vars.NamedVariables {
		namedTensors[k] = v.MustShallowClone()
	}

	return namedTensors
}

// Root gets the root path for this var-store
//
// NOTE: Variables are named and organized using paths. This function returns
// the top level path for the var store and can be combined with '/'
// to create sub-paths.
func (vs *VarStore) Root() *Path {
	return &Path{
		path:     []string{},
		varstore: vs,
		group:    0,
	}
}

// Save saves the var-store variable values to a file
//
// NOTE: Weight values for all the tensors currently stored in the
// var-store gets saved in the given file.
func (vs *VarStore) Save(filepath string) error {
	vs.Vars.mutex.Lock()
	defer vs.Vars.mutex.Unlock()

	// Convert map to []NamedTensor
	var namedTensors []ts.NamedTensor
	for k, v := range vs.Vars.NamedVariables {
		namedTensors = append(namedTensors, ts.NamedTensor{
			Name:   k,
			Tensor: v,
		})
	}

	// return ts.SaveMulti(namedTensors, filepath)
	return ts.SaveMultiNew(namedTensors, filepath)
}

// Load loads the var-store variable values from a file.
//
// NOTE: Weight values for all the tensors currently stored in the
// var-store gets loaded from the given file. Note that the set of
// variables stored in the var-store is not changed, only the values
// for these tensors are modified.
// It will throw error if name of the loaded tensors can not find
// in the current var-store named tensors set.
func (vs *VarStore) Load(filepath string) error {
	namedTensors, err := ts.LoadMultiWithDevice(filepath, vs.device)
	if err != nil {
		return err
	}

	var namedTensorsMap map[string]*ts.Tensor = make(map[string]*ts.Tensor, 0)
	for _, namedTensor := range namedTensors {
		namedTensorsMap[namedTensor.Name] = namedTensor.Tensor
	}

	// Match and in-place copy value (update) from newly loaded tensors
	// to existing named tensors if name is matched. Throw error otherwise.
	vs.Vars.mutex.Lock()
	defer vs.Vars.mutex.Unlock()

	// for tsName, _ := range vs.Vars.NamedVariables {
	for tsName := range vs.Vars.NamedVariables {

		// missing variable
		currTs, ok := namedTensorsMap[tsName]
		if !ok {
			err = fmt.Errorf("Cannot find tensor with name: %v in variable store. \n", tsName)
			return err
		}

		// mismatched shape
		sourceShape := currTs.MustSize()
		destShape := vs.Vars.NamedVariables[tsName].MustSize()
		if !reflect.DeepEqual(destShape, sourceShape) {
			err = fmt.Errorf("Mismatched shape error for variable name: %v - At store: %v - At source %v\n", tsName, destShape, sourceShape)
			return err
		}

		ts.NoGrad(func() {
			vs.Vars.NamedVariables[tsName].Copy_(currTs)
		})
	}
	return nil
}

// LoadPartial loads the var-store variable values from a file if it exists.
//
// Weight values for the tensors currently stored in the var-store and the given file get
// loaded from the given file. If a variable in the var store is not present in the given file,
// it is skipped and its values are not updated. This method should be used if pre-trained
// weight for only parts of the model are available.
// Note that the set of variables stored in the var-store is not changed, only the values
// for these tensors are modified.
//
// Returns a String Vector containing the names of missing variables.
func (vs *VarStore) LoadPartial(filepath string) ([]string, error) {

	namedTensors, err := ts.LoadMultiWithDevice(filepath, vs.device)
	if err != nil {
		return nil, err
	}

	var namedTensorsMap map[string]*ts.Tensor = make(map[string]*ts.Tensor, 0)
	for _, namedTensor := range namedTensors {
		namedTensorsMap[namedTensor.Name] = namedTensor.Tensor
	}

	var missingVariables []string

	// Match and in-place copy value (update) from newly loaded tensors
	// to existing named tensors if name is matched. Throw error otherwise.
	vs.Vars.mutex.Lock()
	defer vs.Vars.mutex.Unlock()

	for tsName := range vs.Vars.NamedVariables {
		var currTs *ts.Tensor
		var ok bool

		// missing variable
		if currTs, ok = namedTensorsMap[tsName]; !ok {
			missingVariables = append(missingVariables, tsName)
			continue
		}

		// mismatched shape
		destShape := currTs.MustSize()
		sourceShape := vs.Vars.NamedVariables[tsName].MustSize()
		if !reflect.DeepEqual(destShape, sourceShape) {
			fmt.Printf("WARNING: Mismatched shape error for variable name: %v - At store: %v - At source %v. Skip loading this weight...\n", tsName, destShape, sourceShape)
			missingVariables = append(missingVariables, tsName)
			continue
		}

		ts.NoGrad(func() {
			vs.Vars.NamedVariables[tsName].Copy_(currTs)
		})
	}

	return missingVariables, nil
}

// Freeze freezes a var store.
//
// Gradients for the variables in this store are not tracked
// anymore.
func (vs *VarStore) Freeze() {
	vs.Vars.mutex.Lock()
	defer vs.Vars.mutex.Unlock()

	for _, v := range vs.Vars.TrainableVariables {
		_, err := v.Tensor.SetRequiresGrad(false, false)
		if err != nil {
			log.Fatalf("Freeze() Error: %v\n", err)
		}
	}
}

// Unfreeze unfreezes a var store.
//
// Gradients for the variables in this store are tracked again.
func (vs *VarStore) Unfreeze() {
	vs.Vars.mutex.Lock()
	defer vs.Vars.mutex.Unlock()

	for _, v := range vs.Vars.TrainableVariables {
		_, err := v.Tensor.SetRequiresGrad(true, false)
		if err != nil {
			log.Fatalf("Unfreeze() Error: %v\n", err)
		}
	}
}

// Copy copies variable values from a source var store to this var store.
//
// All the variables in this var store have to exist with the same
// name in the source var store, otherwise an error is returned.
func (vs *VarStore) Copy(src VarStore) error {
	vs.Vars.mutex.Lock()
	defer vs.Vars.mutex.Unlock()
	src.Vars.mutex.Lock()
	defer src.Vars.mutex.Unlock()

	srcNamedVariables := src.Vars.NamedVariables
	device := vs.device

	for k := range vs.Vars.NamedVariables {
		if _, ok := srcNamedVariables[k]; !ok {
			err := fmt.Errorf("VarStore copy error: cannot find %v in the source var store.\n", k)
			return err
		}
	}

	for k, v := range vs.Vars.NamedVariables {
		srcTs, _ := srcNamedVariables[k]
		srcDevTs, err := srcTs.To(device, false)
		if err != nil {
			return err
		}
		ts.NoGrad(func() {
			ts.Copy_(v, srcDevTs)
		})
	}

	return nil
}

// Path methods:
// =============

// Sub gets a sub-path of the given path.
func (p *Path) Sub(str string) *Path {

	if strings.Contains(str, SEP) {
		log.Fatalf("Sub name cannot contain %v (%v)\n", SEP, str)
	}

	path := p.path
	path = append(path, str)
	return &Path{
		path:     path,
		varstore: p.varstore,
		group:    p.group,
	}
}

// Device gets the device where the var-store variables are stored.
func (p *Path) Device() gotch.Device {

	return p.varstore.device
}

// NOTE: Cannot name as `path` as having a field name `path`
func (p *Path) getpath(name string) string {

	if strings.Contains(name, SEP) {
		log.Fatalf("Sub name cannot contain %v (%v)\n", SEP, name)
	}

	if len(p.path) == 0 {
		return name
	} else {
		return fmt.Sprintf("%v%v%v", strings.Join(p.path, SEP), SEP, name)
	}
}

func (p *Path) add(name string, newTs *ts.Tensor, trainable bool) *ts.Tensor {
	path := p.getpath(name)

	p.varstore.Vars.mutex.Lock()
	defer p.varstore.Vars.mutex.Unlock()

	if _, ok := p.varstore.Vars.NamedVariables[path]; ok {
		path = fmt.Sprintf("%v__%v", path, len(p.varstore.Vars.NamedVariables))
	}

	var (
		tensor *ts.Tensor
		err    error
	)
	if trainable {
		tensor, err = newTs.MustShallowClone().SetRequiresGrad(true, false)
		if err != nil {
			log.Fatalf("Path 'add' method error: %v\n", err)
		}
	} else {
		tensor = newTs.MustShallowClone()
	}

	if trainable {
		v := Var{
			Tensor: tensor,
			Group:  p.group,
		}
		p.varstore.Vars.TrainableVariables = append(p.varstore.Vars.TrainableVariables, v)
	}

	p.varstore.Vars.NamedVariables[path] = tensor

	return tensor
}

// Add adds a tensor to a given path.
func (p *Path) Add(name string, x *ts.Tensor, trainable bool) *ts.Tensor {
	return p.add(name, x, trainable)
}

func (p *Path) getOrAddWithLock(name string, tensor *ts.Tensor, trainable bool, variables Variables) *ts.Tensor {
	path := p.getpath(name)

	// if found, return it
	if v, ok := variables.NamedVariables[path]; ok {
		return v
	}

	// not found, add it
	var err error
	var ttensor *ts.Tensor
	if trainable {
		ttensor, err = tensor.SetRequiresGrad(true, false)
		if err != nil {
			log.Fatalf("Path - call method 'getOrAddWithLock' error: %v\n", err)
		}
	} else {
		ttensor = tensor
	}

	if trainable {
		v := Var{
			Tensor: ttensor,
			Group:  p.group,
		}
		variables.TrainableVariables = append(variables.TrainableVariables, v)
	}

	variables.NamedVariables[path] = ttensor

	return ttensor
}

func (p *Path) SetGroup(g uint) {
	p.group = g
}

// ZerosNoTrain creates a new variable initialized with zeros.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable will not be trainable so
// gradients will not be tracked.
// The variable uses a float tensor initialized with zeros.
func (p *Path) ZerosNoTrain(name string, dims []int64) *ts.Tensor {

	device := p.Device()
	z, err := ts.Zeros(dims, gotch.Float, device)
	if err != nil {
		log.Fatalf("Path - 'ZerosNoTrain' method call error: %v\n", err)
	}

	return p.add(name, z, false)
}

// OnesNoTrain creates a new variable initialized with ones.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable will not be trainable so
// gradients will not be tracked.
// The variable uses a float tensor initialized with ones.
func (p *Path) OnesNoTrain(name string, dims []int64) *ts.Tensor {

	device := p.Device()
	z, err := ts.Ones(dims, gotch.Float, device)
	if err != nil {
		log.Fatalf("Path - 'OnesNoTrain' method call error: %v\n", err)
	}

	return p.add(name, z, false)
}

// NewVar creates a new variable.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized as per the
// related argument.
func (p *Path) NewVar(name string, dims []int64, ini Init) *ts.Tensor {

	v := ini.InitTensor(dims, p.varstore.device)

	return p.add(name, v, true)
}

// Zeros creates a new variable initialized with zeros.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized with zeros.
func (p *Path) Zeros(name string, dims []int64) *ts.Tensor {

	return p.NewVar(name, dims, NewConstInit(0.0))
}

// Ones creates a new variable initialized with ones.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized with ones.
func (p *Path) Ones(name string, dims []int64) *ts.Tensor {

	return p.NewVar(name, dims, NewConstInit(1.0))
}

// RandnStandard creates a new variable initialized randomly with normal distribution.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized randomly using a
// standard normal distribution.
func (p *Path) RandnStandard(name string, dims []int64) *ts.Tensor {

	return p.NewVar(name, dims, NewRandnInit(0.0, 1.0))
}

// Randn creates a new variable initialized randomly with normal distribution.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized randomly using a
// normal distribution with the specified mean and standard deviation.
func (p *Path) Randn(name string, dims []int64, mean float64, stdev float64) *ts.Tensor {

	return p.NewVar(name, dims, NewRandnInit(mean, stdev))
}

// Uniform creates a new variable initialized randomly with uniform distribution.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized randomly using a
// uniform distribution between the specified bounds.
func (p *Path) Uniform(name string, dims []int64, lo, up float64) *ts.Tensor {

	return p.NewVar(name, dims, NewUniformInit(lo, up))
}

// KaimingUniform creates a new variable initialized randomly with kaiming uniform.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized randomly using a
// uniform distribution which bounds follow Kaiming initialization.
func (p *Path) KaimingUniform(name string, dims []int64) *ts.Tensor {

	return p.NewVar(name, dims, NewKaimingUniformInit())
}

// VarCopy creates a new variable initialized by copying an existing tensor.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized by copying some
// given tensor.
func (p *Path) VarCopy(name string, t *ts.Tensor) *ts.Tensor {

	size, err := t.Size()
	if err != nil {
		log.Fatalf("Path - VarCopy method call error: %v\n", err)
	}
	v := p.Zeros(name, size)

	ts.NoGrad(func() {
		ts.Copy_(v, t)
	})

	return v
}

// Get gets the tensor corresponding to a given name if present.
func (p *Path) Get(name string) (*ts.Tensor, error) {

	p.varstore.Vars.mutex.Lock()
	defer p.varstore.Vars.mutex.Unlock()

	v, ok := p.varstore.Vars.NamedVariables[name]
	if !ok {
		err := fmt.Errorf("Path - Get method call error: Cannot find variable for name: %v\n", name)
		return nil, err
	}

	return v.ShallowClone()
}

// Entry gets the entry corresponding to a given name for in-place manipulation.
func (p *Path) Entry(name string) *Entry {
	p.varstore.Vars.mutex.Lock()
	defer p.varstore.Vars.mutex.Unlock()

	return &Entry{
		name:      name,
		variables: &p.varstore.Vars,
		path:      p,
	}
}

// Entry methods:
// ==============

// OrVar returns the existing entry if, otherwise create a new variable.
//
// If this entry name matches the name of a variables stored in the
// var store, the corresponding tensor is returned. Otherwise a new
// variable is added to the var-store with the entry name and is
// initialized according to the init parameter.
func (e *Entry) OrVar(dims []int64, init Init) *ts.Tensor {

	v := init.InitTensor(dims, e.path.varstore.device)
	return e.path.getOrAddWithLock(e.name, v, true, *e.variables)
}

// Returns the existing entry if, otherwise create a new variable.
func (e *Entry) OrVarCopy(tensor *ts.Tensor) *ts.Tensor {

	size, err := tensor.Size()
	if err != nil {
		log.Fatalf("Entry - OrVarCopy method call error: %v\n", err)
	}
	v := e.OrZeros(size)

	ts.NoGrad(func() {
		ts.Copy_(v, tensor)
	})

	return v
}

// Returns the existing entry if, otherwise create a new variable.
func (e *Entry) OrKaimingUniform(dims []int64) *ts.Tensor {

	return e.OrVar(dims, NewKaimingUniformInit())
}

// OrOnes returns the existing entry if, otherwise create a new variable.
func (e *Entry) OrOnes(dims []int64) *ts.Tensor {

	return e.OrVar(dims, NewConstInit(1.0))
}

// OrOnesNoTrain returns the existing entry if, otherwise create a new variable.
func (e *Entry) OrOnesNoTrain(dims []int64) *ts.Tensor {

	o := ts.MustOnes(dims, gotch.Float, e.path.Device())
	return e.path.getOrAddWithLock(e.name, o, true, *e.variables)
}

// OrRandn returns the existing entry if, otherwise create a new variable.
func (e *Entry) OrRandn(dims []int64, mean, stdev float64) *ts.Tensor {

	return e.OrVar(dims, NewRandnInit(mean, stdev))
}

// OrRandnStandard returns the existing entry if, otherwise create a new variable.
func (e *Entry) OrRandnStandard(dims []int64) *ts.Tensor {

	return e.OrVar(dims, NewRandnInit(0.0, 1.0))
}

// OrUniform returns the existing entry if, otherwise create a new variable.
func (e *Entry) OrUniform(dims []int64, lo, up float64) (retVal *ts.Tensor) {

	return e.OrVar(dims, NewUniformInit(lo, up))
}

// OrZeros returns the existing entry if, otherwise create a new variable.
func (e *Entry) OrZeros(dims []int64) *ts.Tensor {

	return e.OrVar(dims, NewConstInit(0.0))
}

// OrZerosNoTrain returns the existing entry if, otherwise create a new variable.
func (e *Entry) OrZerosNoTrain(dims []int64) *ts.Tensor {

	z := ts.MustZeros(dims, gotch.Float, e.path.Device())
	return e.path.getOrAddWithLock(e.name, z, true, *e.variables)
}
