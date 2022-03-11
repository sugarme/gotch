package nn

import (
	"fmt"
	"log"
	"reflect"
	"sort"
	"strings"
	"sync"

	"github.com/sugarme/gotch"
	ts "github.com/sugarme/gotch/tensor"
)

// SEP is a separator to separate path elements in the tensor names.
const SEP = "."

type Var struct {
	Tensor    *ts.Tensor
	Group     uint   // optimizer parameter group
	Type      string // can be "parameter" or "buffer"
	Trainable bool   // marked this variable is either trainable or not.For "buffer" type, it's always `false`
	Persitent bool   // only applied to "buffer" type. All parameters are persistent (when do VarStore.Save()).
}

// VarStore is used to store variables used by one or multiple layers.
// It specifies a SINGLE device where all variables are stored.
type VarStore struct {
	sync.Mutex
	device gotch.Device
	vars   map[string]Var
}

// Path is variable store with an associated path for variables naming.
type Path struct {
	path     []string
	varstore *VarStore
	group    uint // optimizer parameter group
}

// Entry holds an entry corresponding to a given name in Path.
type Entry struct {
	name string
	path *Path
}

// NewVarStore creates a new variable store located on the specified device
func NewVarStore(device gotch.Device) *VarStore {
	return &VarStore{
		device: device,
		vars:   make(map[string]Var, 0),
	}
}

// NOTE:
// To get (initiate) a path, call vs.Root()

// VarStore methods:
// =================

// Device returns device for this VarStore.
func (vs *VarStore) Device() gotch.Device {
	return vs.device
}

// Len returns the number of tensors currently kept in this VarStore.
func (vs *VarStore) Len() int {
	vs.Lock()
	defer vs.Unlock()
	return len(vs.vars)
}

// IsEmpty returns true if no tensors currently kept in this VarStore.
func (vs *VarStore) IsEmpty() bool {
	vs.Lock()
	defer vs.Unlock()
	return (len(vs.vars) == 0)
}

// TrainableVariabless returns reference to all trainable variables kept in VarStore.
func (vs *VarStore) TrainableVariables() []ts.Tensor {
	vs.Lock()
	defer vs.Unlock()

	var trainables []ts.Tensor
	for _, v := range vs.vars {
		x := v.Tensor
		if x.MustRequiresGrad() {
			trainables = append(trainables, *x)
		}
	}

	return trainables
}

// Variables returns reference of all variables and their names in a map[variable_name]Tensor
//
// NOTE. returned map includes all variables of "parameter" and "buffer" type.
func (vs *VarStore) Variables() map[string]ts.Tensor {
	vs.Lock()
	defer vs.Unlock()

	namedTensors := make(map[string]ts.Tensor, 0)
	for k, v := range vs.vars {
		namedTensors[k] = *v.Tensor
	}

	return namedTensors
}

// Root gets the root path for this VarStore.
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

// Save saves the VarStore variable values to a file.
//
// NOTE: Weight values for all the tensors currently stored in the
// var-store gets saved in the given file.
func (vs *VarStore) Save(filepath string) error {
	vs.Lock()
	defer vs.Unlock()

	var namedTensors []ts.NamedTensor
	for k, v := range vs.vars {
		if v.Type == "parameter" || (v.Type == "buffer" && v.Persitent) {
			namedTensors = append(namedTensors, ts.NamedTensor{
				Name:   k,
				Tensor: v.Tensor,
			})
		}
	}

	// return ts.SaveMulti(namedTensors, filepath)
	return ts.SaveMultiNew(namedTensors, filepath)
}

// Load loads VarStore variable values from a file.
//
// NOTE: Weight values for all the tensors currently stored in the
// VarStore gets loaded from the given file. Note that the set of
// variables stored in the VarStore is not changed, only the values
// for these tensors are modified.
// It will throw error if name of the loaded tensors can not find
// in the current VarStore named tensors set.
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
	vs.Lock()
	defer vs.Unlock()

	for name, v := range vs.vars {
		// missing variable
		currTs, ok := namedTensorsMap[name]
		if !ok {
			err = fmt.Errorf("VarStore.Load() failed: there's a tensor with name %q in VarStore, but not found in the loaded weights.\n", name)
			return err
		}

		// mismatched shape
		sourceShape := currTs.MustSize()
		destShape := v.Tensor.MustSize()
		if !reflect.DeepEqual(destShape, sourceShape) {
			err = fmt.Errorf("Mismatched shape error for variable name: %v - At store: %v - At source %v\n", name, destShape, sourceShape)
			return err
		}

		ts.NoGrad(func() {
			v.Tensor.Copy_(currTs)
		})
	}
	return nil
}

// LoadPartial loads the VarStore variable values from a file if it exists.
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
	vs.Lock()
	defer vs.Unlock()

	for name, v := range vs.vars {
		var currTs *ts.Tensor
		var ok bool

		// missing variable
		if currTs, ok = namedTensorsMap[name]; !ok {
			missingVariables = append(missingVariables, name)
			continue
		}

		// mismatched shape
		destShape := currTs.MustSize()
		sourceShape := v.Tensor.MustSize()
		if !reflect.DeepEqual(destShape, sourceShape) {
			fmt.Printf("WARNING: Mismatched shape error for variable name: %v - At store: %v - At source %v. Skip loading this weight...\n", name, destShape, sourceShape)
			missingVariables = append(missingVariables, name)
			continue
		}

		ts.NoGrad(func() {
			v.Tensor.Copy_(currTs)
		})
	}

	return missingVariables, nil
}

// Freeze freezes this VarStore.
//
// Gradients for the variables in this store are not tracked anymore.
func (vs *VarStore) Freeze() error {
	vs.Lock()
	defer vs.Unlock()

	for name, v := range vs.vars {
		err := v.Tensor.RequiresGrad_(false)
		if err != nil {
			err = fmt.Errorf("VarStore.Freeze() set 'requiresGrad' for tensor %q failed.", name)
			return err
		}
	}

	return nil
}

// Unfreeze unfreezes a VarStore.
//
// Gradients for the variables in this store are tracked again.
func (vs *VarStore) Unfreeze() error {
	vs.Lock()
	defer vs.Unlock()

	for name, v := range vs.vars {
		if v.Type == "parameter" && v.Trainable {
			err := v.Tensor.RequiresGrad_(true)
			err = fmt.Errorf("VarStore.Freeze() set 'requiresGrad' for tensor %q failed.", name)
			return err
		}
	}
	return nil
}

// Copy copies variable values from a source VarStore to this VarStore.
//
// All the variables in this var store have to exist with the same
// name in the source var store, otherwise an error is returned.
func (vs *VarStore) Copy(src VarStore) error {
	vs.Lock()
	defer vs.Unlock()
	src.Lock()
	defer src.Unlock()

	srcVars := src.vars
	device := vs.device

	for k := range vs.vars {
		if _, ok := srcVars[k]; !ok {
			err := fmt.Errorf("VarStore.Copy() failed: cannot find %q in the source VarStore.\n", k)
			return err
		}
	}

	for k, v := range vs.vars {
		srcV := srcVars[k]
		srcDevTs, err := srcV.Tensor.To(device, false)
		if err != nil {
			return err
		}
		ts.NoGrad(func() {
			v.Tensor.Copy_(srcDevTs)
		})
		srcDevTs.MustDrop()
	}

	return nil
}

// Summary prints a simple list of all named variables with their shapes.
func (vs *VarStore) Summary() {
	vars := vs.vars
	layers := make([]string, 0, len(vars))
	for name := range vars {
		layers = append(layers, name)
	}
	sort.Strings(layers)
	for _, l := range layers {
		var x *ts.Tensor
		var isBuffer bool
		for name, v := range vars {
			if name == l {
				x = v.Tensor
				isBuffer = v.Type == "buffer"
				break
			}
		}
		if isBuffer {
			fmt.Printf("%s - [buffer] - %+v\n", l, x.MustSize())
		} else {
			fmt.Printf("%s - %+v\n", l, x.MustSize())
		}
	}

	fmt.Printf("Num of layers: %v\n", len(vars))
}

// Path methods:
// =============

// Sub gets a sub-path of the given path.
func (p *Path) Sub(str string) *Path {
	if strings.Contains(str, SEP) {
		log.Fatalf("Path.Sub() failed: name cannot contain %v (%v)\n", SEP, str)
	}

	path := p.path
	path = append(path, str)
	return &Path{
		path:     path,
		varstore: p.varstore,
		group:    p.group,
	}
}

// Paths returns all sub paths from current path.
func (p *Path) Paths() []string {
	return p.path
}

// Device gets the device where the VarStore variables are stored.
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

func (p *Path) add(name string, newTs *ts.Tensor, trainable bool, varType string, persistent bool) (*ts.Tensor, error) {
	path := p.getpath(name)

	p.varstore.Lock()
	defer p.varstore.Unlock()

	if _, ok := p.varstore.vars[path]; ok {
		path = fmt.Sprintf("%v__%v", path, len(p.varstore.vars))
	}

	var (
		tensor *ts.Tensor
		err    error
	)
	if trainable {
		tensor, err = newTs.SetRequiresGrad(true, false)
		if err != nil {
			return nil, err
		}
	} else {
		tensor = newTs.MustShallowClone()
	}

	v := Var{
		Tensor:    tensor,
		Group:     p.group,
		Trainable: trainable,
		Type:      varType,
		Persitent: persistent,
	}
	p.varstore.vars[path] = v

	return tensor, nil
}

type AddOpts struct {
	VarType    string
	Persistent bool
}

type AddOpt func(*AddOpts)

func defaultAddOpts() *AddOpts {
	return &AddOpts{
		VarType:    "parameter",
		Persistent: true,
	}
}

func WithVarType(v string) AddOpt {
	if v != "parameter" && v != "buffer" {
		log.Fatalf("WithVarType() failed(): invalid option variable type. Input must be either 'parameter' or 'buffer'.")
	}

	return func(o *AddOpts) {
		o.VarType = v
	}
}

func WithPersistent(v bool) AddOpt {
	return func(o *AddOpts) {
		o.Persistent = v
	}
}

// Add adds a tensor to a given path.
//
// Args
// - name: intention name of variable in VarStore (if duplicated, it will be added a suffix number)
// - x: tensor holding values to keep in VarStore
// - trainable: marked whether tensor is trainable.
// - o.VarType: variable type, i.e., either "parameter" or "buffer"
// - o.Persistent: whether to save this variables when `VarStore.Save()` is called. Only applied to `buffer` type.
// Returns a reference to a tensor stored in VarStore and error if occurred.
func (p *Path) Add(name string, x *ts.Tensor, trainable bool, opts ...AddOpt) (*ts.Tensor, error) {
	o := defaultAddOpts()
	for _, opt := range opts {
		opt(o)
	}

	return p.add(name, x, trainable, o.VarType, o.Persistent)
}

func (p *Path) getOrAddWithLock(name string, tensor *ts.Tensor, trainable bool, opts ...AddOpt) (*ts.Tensor, error) {
	path := p.getpath(name)

	// if found, return it
	if v, ok := p.varstore.vars[path]; ok {
		return v.Tensor, nil
	}

	// not found, add it
	return p.Add(name, tensor, trainable, opts...)
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
func (p *Path) ZerosNoTrain(name string, dims []int64, opts ...AddOpt) (*ts.Tensor, error) {
	device := p.Device()
	z, err := ts.Zeros(dims, gotch.Float, device)
	if err != nil {
		err = fmt.Errorf("Path.ZerosNoTrain() failed: %w", err)
		return nil, err
	}

	out, err := p.Add(name, z, false, opts...)
	if err != nil {
		return nil, err
	}
	z.MustDrop()

	return out, nil
}

// OnesNoTrain creates a new variable initialized with ones.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable will not be trainable so
// gradients will not be tracked.
// The variable uses a float tensor initialized with ones.
func (p *Path) OnesNoTrain(name string, dims []int64, opts ...AddOpt) (*ts.Tensor, error) {
	device := p.Device()
	z, err := ts.Ones(dims, gotch.Float, device)
	if err != nil {
		err = fmt.Errorf("Path.OneNoTrain() failed: %w", err)
		return nil, err
	}

	out, err := p.Add(name, z, false, opts...)
	if err != nil {
		return nil, err
	}
	z.MustDrop()

	return out, nil
}

// NewVar creates a new variable.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized as per the
// related argument.
func (p *Path) NewVar(name string, dims []int64, ini Init, opts ...AddOpt) (*ts.Tensor, error) {
	v := ini.InitTensor(dims, p.varstore.device)
	out, err := p.Add(name, v, true, opts...)
	if err != nil {
		return nil, err
	}
	v.MustDrop()
	return out, err
}

// MustNewVar create a new variable. It panics if error.
func (p *Path) MustNewVar(name string, dims []int64, ini Init, opts ...AddOpt) *ts.Tensor {
	x, err := p.NewVar(name, dims, ini, opts...)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// Zeros creates a new variable initialized with zeros.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized with zeros.
func (p *Path) Zeros(name string, dims []int64, opts ...AddOpt) (*ts.Tensor, error) {
	return p.NewVar(name, dims, NewConstInit(0.0), opts...)
}

// MustZeros create a new variables with zero values. It panics if error.
func (p *Path) MustZeros(name string, dims []int64, opts ...AddOpt) *ts.Tensor {
	x, err := p.Zeros(name, dims, opts...)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// Ones creates a new variable initialized with ones.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized with ones.
func (p *Path) Ones(name string, dims []int64, opts ...AddOpt) (*ts.Tensor, error) {
	return p.NewVar(name, dims, NewConstInit(1.0), opts...)
}

// MustOnes creates a new variable initialized with ones. It panics if error occurred.
func (p *Path) MustOnes(name string, dims []int64, opts ...AddOpt) *ts.Tensor {
	x, err := p.Ones(name, dims, opts...)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// RandnStandard creates a new variable initialized randomly with normal distribution.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized randomly using a
// standard normal distribution.
func (p *Path) RandnStandard(name string, dims []int64, opts ...AddOpt) (*ts.Tensor, error) {

	return p.NewVar(name, dims, NewRandnInit(0.0, 1.0), opts...)
}

// MustRandnStandard creates a new variable initialized randomly with normal distribution. It panics if error occurred.
func (p *Path) MustRandnStandard(name string, dims []int64, opts ...AddOpt) *ts.Tensor {
	x, err := p.RandnStandard(name, dims, opts...)
	if err != nil {
		log.Fatal(err)
	}

	return x
}

// Randn creates a new variable initialized randomly with normal distribution.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized randomly using a
// normal distribution with the specified mean and standard deviation.
func (p *Path) Randn(name string, dims []int64, mean float64, stdev float64, opts ...AddOpt) (*ts.Tensor, error) {
	return p.NewVar(name, dims, NewRandnInit(mean, stdev), opts...)
}

// MustRandn creates a new variable initialized randomly with normal distribution. It panics if error occurred.
func (p *Path) MustRandn(name string, dims []int64, mean float64, stdev float64, opts ...AddOpt) *ts.Tensor {
	x, err := p.Randn(name, dims, mean, stdev, opts...)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// Uniform creates a new variable initialized randomly with uniform distribution.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized randomly using a
// uniform distribution between the specified bounds.
func (p *Path) Uniform(name string, dims []int64, lo, up float64, opts ...AddOpt) (*ts.Tensor, error) {
	return p.NewVar(name, dims, NewUniformInit(lo, up), opts...)
}

// MustUniform creates a new variable initialized randomly with uniform distribution. It panics if error occurred.
func (p *Path) MustUniform(name string, dims []int64, lo, up float64, opts ...AddOpt) *ts.Tensor {
	x, err := p.Uniform(name, dims, lo, up, opts...)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// KaimingUniform creates a new variable initialized randomly with kaiming uniform.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized randomly using a
// uniform distribution which bounds follow Kaiming initialization.
func (p *Path) KaimingUniform(name string, dims []int64, opts ...AddOpt) (*ts.Tensor, error) {
	return p.NewVar(name, dims, NewKaimingUniformInit(), opts...)
}

// MustKaimingUniform creates a new variable initialized randomly with kaiming uniforms. It panics if error occurred.
func (p *Path) MustKaimingUniform(name string, dims []int64, opts ...AddOpt) *ts.Tensor {
	x, err := p.KaimingUniform(name, dims, opts...)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// VarCopy creates a new variable initialized by copying an existing tensor.
//
// The new variable is named according to the name parameter and
// has the specified shape. The variable is trainable, its gradient
// will be tracked.
// The variable uses a float tensor initialized by copying some
// given tensor.
func (p *Path) VarCopy(name string, t *ts.Tensor) (*ts.Tensor, error) {
	size, err := t.Size()
	if err != nil {
		err = fmt.Errorf("Path.VarCopy() failed: %w\n", err)
		return nil, err
	}
	v, err := p.Zeros(name, size)
	if err != nil {
		return nil, err
	}

	ts.NoGrad(func() {
		ts.Copy_(v, t)
	})

	return v, nil
}

// VarCopy creates a new variable initialized by copying an existing tensor.
func (p *Path) MustVarCopy(name string, t *ts.Tensor) *ts.Tensor {
	x, err := p.VarCopy(name, t)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// Get gets a reference to tensor corresponding to a given name if present.
func (p *Path) Get(name string) (*ts.Tensor, error) {
	p.varstore.Lock()
	defer p.varstore.Unlock()

	v, ok := p.varstore.vars[name]
	if !ok {
		err := fmt.Errorf("Path.Get() failed: Cannot find variable for name: %v\n", name)
		return nil, err
	}

	return v.Tensor, nil
}

// MustGet gets a reference to a tensor corresponding to a given name if present. It panics if error occurred.
func (p *Path) MustGet(name string) *ts.Tensor {
	x, err := p.Get(name)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// Entry gets the entry corresponding to a given name for in-place manipulation.
func (p *Path) Entry(name string) *Entry {
	p.varstore.Lock()
	defer p.varstore.Unlock()

	return &Entry{
		name: name,
		path: p,
	}
}

// Entry methods:
// ==============

// OrVar returns the existing entry if found, otherwise create a new variable.
//
// If this entry name matches the name of a variables stored in the
// var store, the corresponding tensor is returned. Otherwise a new
// variable is added to the var-store with the entry name and is
// initialized according to the init parameter.
func (e *Entry) OrVar(dims []int64, init Init, opts ...AddOpt) (*ts.Tensor, error) {
	v := init.InitTensor(dims, e.path.varstore.device)
	out, err := e.path.getOrAddWithLock(e.name, v, true, opts...)
	if err != nil {
		return nil, err
	}
	v.MustDrop()

	return out, nil
}

// MustOrVar returns the existing entry if found, otherwise creates a new variable. It panics if error.
func (e *Entry) MustOrVar(dims []int64, init Init, opts ...AddOpt) *ts.Tensor {
	x, err := e.OrVar(dims, init, opts...)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// OrZeros returns the existing entry if found, otherwise creates a new variable.
func (e *Entry) OrZeros(dims []int64, opts ...AddOpt) (*ts.Tensor, error) {
	return e.OrVar(dims, NewConstInit(0.0))
}

// MustOrZeros returns the exising entry if found, otherwise creates a new variable.
func (e *Entry) MustOrZeros(dims []int64, opts ...AddOpt) *ts.Tensor {
	x, err := e.OrZeros(dims, opts...)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// OrVarCopy returns the existing entry if found, otherwise create a new variable.
func (e *Entry) OrVarCopy(tensor *ts.Tensor) (*ts.Tensor, error) {
	size, err := tensor.Size()
	if err != nil {
		return nil, err
	}
	v, err := e.OrZeros(size)
	if err != nil {
		return nil, err
	}

	ts.NoGrad(func() {
		ts.Copy_(v, tensor)
	})

	return v, nil
}

// MustOrVarCopy returns the existing entry if found, otherwise create a new variable.
func (e *Entry) MustOrVarCopy(tensor *ts.Tensor) *ts.Tensor {
	x, err := e.OrVarCopy(tensor)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// OrKaimingUniform returns the existing entry if, otherwise create a new variable.
func (e *Entry) OrKaimingUniform(dims []int64, opts ...AddOpt) (*ts.Tensor, error) {
	return e.OrVar(dims, NewKaimingUniformInit(), opts...)
}

// MustOrKaimingUniform returns the existing entry if, otherwise create a new variable.
func (e *Entry) MustOrKaimingUniform(dims []int64, opts ...AddOpt) *ts.Tensor {
	x, err := e.OrKaimingUniform(dims, opts...)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// OrOnes returns the existing entry if found, otherwise create a new variable.
func (e *Entry) OrOnes(dims []int64, opts ...AddOpt) (*ts.Tensor, error) {
	return e.OrVar(dims, NewConstInit(1.0), opts...)
}

// MustOrOnes returns the existing entry if found, otherwise create a new variable.
func (e *Entry) MustOrOnes(dims []int64, opts ...AddOpt) *ts.Tensor {
	x, err := e.OrOnes(dims, opts...)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// OrOnesNoTrain returns the existing entry if found, otherwise create a new variable.
func (e *Entry) OrOnesNoTrain(dims []int64, opts ...AddOpt) (*ts.Tensor, error) {
	o := ts.MustOnes(dims, gotch.Float, e.path.Device())
	out, err := e.path.getOrAddWithLock(e.name, o, true, opts...)
	if err != nil {
		return nil, err
	}
	o.MustDrop()

	return out, nil
}

// MustOrOnesNoTrain returns the existing entry if found, otherwise create a new variable.
func (e *Entry) MustOrOnesNoTrain(dims []int64, opts ...AddOpt) *ts.Tensor {
	x, err := e.OrOnesNoTrain(dims, opts...)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// OrRandn returns the existing entry if found, otherwise create a new variable.
func (e *Entry) OrRandn(dims []int64, mean, stdev float64, opts ...AddOpt) (*ts.Tensor, error) {
	return e.OrVar(dims, NewRandnInit(mean, stdev), opts...)
}

// MustOrRandn returns the existing entry if, otherwise create a new variable.
func (e *Entry) MustOrRandn(dims []int64, mean, stdev float64, opts ...AddOpt) *ts.Tensor {
	x, err := e.OrRandn(dims, mean, stdev, opts...)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// OrRandnStandard returns the existing entry if found, otherwise create a new variable.
func (e *Entry) OrRandnStandard(dims []int64, opts ...AddOpt) (*ts.Tensor, error) {
	return e.OrVar(dims, NewRandnInit(0.0, 1.0), opts...)
}

// MustOrRandnStandard returns the existing entry if, otherwise create a new variable.
func (e *Entry) MustOrRandnStandard(dims []int64, opts ...AddOpt) *ts.Tensor {
	x, err := e.OrRandnStandard(dims, opts...)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// OrUniform returns the existing entry if found, otherwise create a new variable.
func (e *Entry) OrUniform(dims []int64, lo, up float64, opts ...AddOpt) (*ts.Tensor, error) {
	return e.OrVar(dims, NewUniformInit(lo, up), opts...)
}

// MustOrUniform returns the existing entry if found, otherwise create a new variable.
func (e *Entry) MustOrUniform(dims []int64, lo, up float64, opts ...AddOpt) *ts.Tensor {
	x, err := e.OrUniform(dims, lo, up, opts...)
	if err != nil {
		log.Fatal(err)
	}
	return x
}

// OrZerosNoTrain returns the existing entry if found, otherwise create a new variable.
func (e *Entry) OrZerosNoTrain(dims []int64, opts ...AddOpt) (*ts.Tensor, error) {
	z := ts.MustZeros(dims, gotch.Float, e.path.Device())
	out, err := e.path.getOrAddWithLock(e.name, z, true, opts...)
	if err != nil {
		return nil, err
	}
	z.MustDrop()

	return out, nil
}

// MustOrZerosNoTrain returns the existing entry if found, otherwise create a new variable.
func (e *Entry) MustOrZerosNoTrain(dims []int64, opts ...AddOpt) *ts.Tensor {
	x, err := e.OrZerosNoTrain(dims, opts...)
	if err != nil {
		log.Fatal(err)
	}
	return x
}
