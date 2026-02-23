package ts_test

import (
	"testing"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/ts"
)

// TestGradientStatePersistence verifies that gradient state persists correctly
// This tests for PyTorch 2.10.0 gradient state management issues
func TestGradientStatePersistence(t *testing.T) {
	// Enable gradients
	prev1 := ts.MustGradSetEnabled(true)
	t.Logf("First enable: previous state was %v", prev1)

	// Check if it stayed enabled
	prev2 := ts.MustGradSetEnabled(true)
	t.Logf("Second enable: previous state was %v", prev2)

	if !prev2 {
		t.Errorf("Gradient state did not persist: set to true, but immediately became false")
	}

	// Disable gradients
	prev3 := ts.MustGradSetEnabled(false)
	t.Logf("Disable: previous state was %v", prev3)

	if !prev3 {
		t.Errorf("Expected gradients to be enabled before disabling, but they were already disabled")
	}

	// Check if it stayed disabled
	prev4 := ts.MustGradSetEnabled(false)
	t.Logf("Second disable: previous state was %v", prev4)

	if prev4 {
		t.Errorf("Gradient state did not persist: set to false, but immediately became true")
	}

	// Re-enable for other tests
	ts.MustGradSetEnabled(true)
}

// TestGradientStateAfterOperations tests if operations affect gradient state
func TestGradientStateAfterOperations(t *testing.T) {
	// Start with gradients enabled
	ts.MustGradSetEnabled(true)

	// Create tensor
	x := ts.MustRandn([]int64{2, 2}, gotch.Float, gotch.CPU)
	defer x.MustDrop()

	// Check state after tensor creation
	stateAfterCreate := ts.MustGradSetEnabled(true)
	t.Logf("After tensor creation: grads were %v", stateAfterCreate)

	if !stateAfterCreate {
		t.Errorf("Gradient state changed after tensor creation")
	}

	// Drop tensor
	x.MustDrop()

	// Check state after drop
	stateAfterDrop := ts.MustGradSetEnabled(true)
	t.Logf("After tensor drop: grads were %v", stateAfterDrop)

	if !stateAfterDrop {
		t.Errorf("Gradient state changed after tensor drop")
	}
}
