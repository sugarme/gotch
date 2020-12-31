package dutil_test

import (
	"testing"

	"github.com/sugarme/gotch/dutil"
)

func TestNewKFold(t *testing.T) {
	// invalid kfold
	n := 10
	nfolds := 11

	_, err := dutil.NewKFold(n, dutil.WithNFolds(nfolds), dutil.WithKFoldShuffle(true))
	if err == nil {
		t.Errorf("Expected error: invalid number of folds. Got nil.")
	}

	// valid
	nfolds = 3
	_, err = dutil.NewKFold(n, dutil.WithNFolds(nfolds), dutil.WithKFoldShuffle(true))
	if err != nil {
		t.Errorf("Unexpected error. Got %v.\n", err)
	}
}

func TestKFold_Split(t *testing.T) {
	n := 11
	nfolds := 3
	trainLen := 6
	testLen := 3

	kf, err := dutil.NewKFold(n, dutil.WithNFolds(nfolds), dutil.WithKFoldShuffle(true))
	if err != nil {
		t.Error(err)
	}

	splits := kf.Split()

	if len(splits) != nfolds {
		t.Errorf("Want number of folds: %v\n", nfolds)
		t.Errorf("Got number of folds: %v\n", len(splits))
	}

	for _, f := range splits {
		if len(f.Train) != trainLen {
			t.Errorf("Expect train length: %v\n", trainLen)
			t.Errorf("Got train length: %v\n", len(f.Train))
		}

		if len(f.Test) != testLen {
			t.Errorf("Expect test length: %v\n", testLen)
			t.Errorf("Got test length: %v\n", len(f.Test))
		}
	}
}
