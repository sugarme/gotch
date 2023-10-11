package nn

import (
	"fmt"
	"testing"
	"time"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

// Test whether InitTensor() can cause memory blow-up due to accumulate gradient.
func TestInitTensor_Memcheck(t *testing.T) {
	gotch.PrintMemStats("Start")
	device := gotch.CPU
	vs := NewVarStore(device)
	params := 500

	path := vs.Root()
	dims := []int64{1024, 1024}
	for i := 0; i < params; i++ {
		ts.NoGrad(func() {
			name := fmt.Sprintf("param_%v", i)
			x := ts.MustRandn(dims, gotch.DefaultDType, device)
			path.MustAdd(name, x, false)
			x.MustDrop()
		})
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
