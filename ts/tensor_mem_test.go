package ts

import (
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"testing"
	"time"

	"github.com/sugarme/gotch"
)

var n int = 10

func newData() []float32 {
	n := 3 * 224 * 224 * 12
	data := make([]float32, n)
	for i := 0; i < n; i++ {
		data[i] = rand.Float32()
	}

	return data
}

func printMemStats(message string, rtm runtime.MemStats) {
	fmt.Println("\n===", message, "===")
	fmt.Println("Mallocs: ", rtm.Mallocs)
	fmt.Println("Frees: ", rtm.Frees)
	fmt.Println("LiveObjects: ", rtm.Mallocs-rtm.Frees)
	fmt.Println("PauseTotalNs: ", rtm.PauseTotalNs)
	fmt.Println("NumGC: ", rtm.NumGC)
	fmt.Println("LastGC: ", time.UnixMilli(int64(rtm.LastGC/1_000_000)))
	fmt.Println("HeapObjects: ", rtm.HeapObjects)
	fmt.Println("HeapAlloc: ", rtm.HeapAlloc)
}

func TestMem(t *testing.T) {
	var rtm runtime.MemStats
	runtime.ReadMemStats(&rtm)
	printMemStats("Start", rtm)

	for i := 0; i < n; i++ {
		x := MustOfSlice(newData())
		log.Printf("created tensor : %q\n", x.Name())
	}

	runtime.ReadMemStats(&rtm)
	printMemStats("After completing loop", rtm)

	runtime.GC()
	runtime.ReadMemStats(&rtm)
	printMemStats("After forced GC", rtm)

	fmt.Printf(CheckCMemLeak())
}

func TestMem1(t *testing.T) {
	var rtm runtime.MemStats
	runtime.ReadMemStats(&rtm)
	printMemStats("Start", rtm)

	for i := 0; i < n; i++ {
		x, err := Randn([]int64{2, 3, 224, 224}, gotch.Float, gotch.CPU)
		if err != nil {
			panic(err)
		}

		log.Printf("created tensor : %q\n", x.Name())
		time.Sleep(time.Millisecond * 3000) // 3secs
	}

	CleanUp()

	runtime.ReadMemStats(&rtm)
	printMemStats("After completing loop", rtm)

	runtime.GC()
	runtime.ReadMemStats(&rtm)
	printMemStats("After forced GC", rtm)

	fmt.Printf(CheckCMemLeak())
}
