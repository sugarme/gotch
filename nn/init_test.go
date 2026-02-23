package nn

import (
	"fmt"
	"testing"
	"time"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

// Test whether InitTensor() can cause memory blow-up due to accumulate gradient.
//
// PyTorch 2.10.0 Note: This test does NOT need NoGrad wrapper.
// Parameter initialization doesn't require disabling gradients.
// Previous versions used NoGrad as a workaround for gradient state bugs,
// but after fixing NoGrad() to properly save/restore state, it's unnecessary.
func TestInitTensor_Memcheck(t *testing.T) {
	gotch.PrintMemStats("Start")
	device := gotch.CPU
	vs := NewVarStore(device)
	params := 500

	path := vs.Root()
	dims := []int64{1024, 1024}

	// Create parameters - no NoGrad needed
	for i := 0; i < params; i++ {
		name := fmt.Sprintf("param_%v", i)
		x := ts.MustRandn(dims, gotch.DefaultDType, device)
		path.MustAdd(name, x, false)
		// NOTE: Don't drop x - VarStore owns it after MustAdd
	}

	// vs.Summary()

	fmt.Printf("vs created...\n")
	// printMemStats("After varstore created")

	vs.Destroy()
	ts.CleanUp()

	fmt.Printf("vs deleted...\n")

	// printMemStats("After varstore deleted")

	time.Sleep(time.Second * 10)
	gotch.PrintMemStats("Final")
}
