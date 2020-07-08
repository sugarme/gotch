// A wrapper around the linux syscall sysinfo(2).
package gotch

// helper to debug memory blow-up

import (
	"fmt"
	"sync"
	"syscall"
	"time"
)

// Go-ized http://man7.org/linux/man-pages/man2/sysinfo.2.html
type SI struct {
	Uptime       time.Duration // time since boot
	Loads        [3]float64    // 1, 5, and 15 minute load averages, see e.g. UPTIME(1)
	Procs        uint64        // number of current processes
	TotalRam     uint64        // total usable main memory size [kB]
	FreeRam      uint64        // available memory size [kB]
	SharedRam    uint64        // amount of shared memory [kB]
	BufferRam    uint64        // memory used by buffers [kB]
	TotalSwap    uint64        // total swap space size [kB]
	FreeSwap     uint64        // swap space still available [kB]
	TotalHighRam uint64        // total high memory size [kB]
	FreeHighRam  uint64        // available high memory size [kB]
	mu           sync.Mutex    // ensures atomic writes; protects the following fields
}

var sis = &SI{}

// Get the linux sysinfo data structure.
//
// Useful links in the wild web:
// http://man7.org/linux/man-pages/man2/sysinfo.2.html
// http://man7.org/linux/man-pages/man1/uptime.1.html
// https://github.com/capnm/golang/blob/go1.1.1/src/pkg/syscall/zsyscall_linux_amd64.go#L1050
// https://github.com/capnm/golang/blob/go1.1.1/src/pkg/syscall/ztypes_linux_amd64.go#L528
// https://github.com/capnm/golang/blob/go1.1.1/src/pkg/syscall/ztypes_linux_arm.go#L502
func GetSysInfo() *SI {

	/*
	   // Note: uint64 is uint32 on 32 bit CPUs
	   type Sysinfo_t struct {
	   	Uptime    int64		// Seconds since boot
	   	Loads     [3]uint64	// 1, 5, and 15 minute load averages
	   	Totalram  uint64	// Total usable main memory size
	   	Freeram   uint64	// Available memory size
	   	Sharedram uint64	// Amount of shared memory
	   	Bufferram uint64	// Memory used by buffers
	   	Totalswap uint64	// Total swap space size
	   	Freeswap  uint64	// swap space still available
	   	Procs     uint16	// Number of current processes
	   	Pad       uint16
	   	Pad_cgo_0 [4]byte
	   	Totalhigh uint64	// Total high memory size
	   	Freehigh  uint64	// Available high memory size
	   	Unit      uint32	// Memory unit size in bytes
	   	X_f       [0]byte
	   	Pad_cgo_1 [4]byte	// Padding to 64 bytes
	   }
	*/

	// ~1kB garbage
	si := &syscall.Sysinfo_t{}

	// XXX is a raw syscall thread safe?
	err := syscall.Sysinfo(si)
	if err != nil {
		panic("Commander, we have a problem. syscall.Sysinfo:" + err.Error())
	}
	scale := 65536.0 // magic

	defer sis.mu.Unlock()
	sis.mu.Lock()

	unit := uint64(si.Unit) * 1024 // kB

	sis.Uptime = time.Duration(si.Uptime) * time.Second
	sis.Loads[0] = float64(si.Loads[0]) / scale
	sis.Loads[1] = float64(si.Loads[1]) / scale
	sis.Loads[2] = float64(si.Loads[2]) / scale
	sis.Procs = uint64(si.Procs)

	sis.TotalRam = uint64(si.Totalram) / unit
	sis.FreeRam = uint64(si.Freeram) / unit
	sis.BufferRam = uint64(si.Bufferram) / unit
	sis.TotalSwap = uint64(si.Totalswap) / unit
	sis.FreeSwap = uint64(si.Freeswap) / unit
	sis.TotalHighRam = uint64(si.Totalhigh) / unit
	sis.FreeHighRam = uint64(si.Freehigh) / unit

	return sis
}

// Make the "fmt" Stringer interface happy.
func (si SI) String() string {
	// XXX: Is the copy of SI done atomic? Not sure.
	// Without an outer lock this may print a junk.
	return fmt.Sprintf("uptime\t\t%v\nload\t\t%2.2f %2.2f %2.2f\nprocs\t\t%d\n"+
		"ram  total\t%d kB\nram  free\t%d kB\nram  buffer\t%d kB\n"+
		"swap total\t%d kB\nswap free\t%d kB",
		//"high ram total\t%d kB\nhigh ram free\t%d kB\n"
		si.Uptime, si.Loads[0], si.Loads[1], si.Loads[2], si.Procs,
		si.TotalRam, si.FreeRam, si.BufferRam,
		si.TotalSwap, si.FreeSwap,
		// archaic si.TotalHighRam, si.FreeHighRam
	)
}

/*
Convert to string in a thread safe way.
 Output:
	uptime		279h6m21s
	load		0.12 0.04 0.05
	procs		143
	ram  total	383752 kB
	ram  free	254980 kB
	ram  buffer	7640 kB
	swap total	887800 kB
	swap free	879356 kB
*/
func (si *SI) ToString() string {
	defer si.mu.Unlock()
	si.mu.Lock()
	return si.String()
}
