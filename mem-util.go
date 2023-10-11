package gotch

// helper to debug memory blow-up

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"text/tabwriter"
)

func PrintMemStats(messageOpt ...string) {
	message := "Memory Stats"
	if len(messageOpt) > 0 {
		message = fmt.Sprintf("%s: %s", message, messageOpt[0])
	}

	var rtm runtime.MemStats
	runtime.ReadMemStats(&rtm)

	tp := newTablePrinter()
	tp.title = message

	tp.AddRecord("|", "Allocated heap objects", padRight(fmt.Sprintf("%v", rtm.Mallocs), 10), "|")
	tp.AddRecord("|", "Released heap objects", padRight(fmt.Sprintf("%v", rtm.Frees), 10), "|")
	tp.AddRecord("|", "Living heap objects", padRight(fmt.Sprintf("%v", rtm.HeapObjects), 10), "|")
	tp.AddRecord("|", "Memory in use by heap objects (bytes)", padRight(fmt.Sprintf("%v", rtm.HeapAlloc), 10), "|")
	tp.AddRecord("|", "Reserved memory (by Go runtime for heap, stack,...) (bytes)", padRight(fmt.Sprintf("%v", rtm.Sys), 10), "|")
	tp.AddRecord("|", "Total pause time by GC (nanoseconds)", padRight(fmt.Sprintf("%v", rtm.PauseTotalNs), 10), "|")
	tp.AddRecord("|", "Number of GC called", padRight(fmt.Sprintf("%v", rtm.NumGC), 10), "|")
	// tp.AddRecord("Last GC called", fmt.Sprintf("%v", time.UnixMilli(int64(rtm.LastGC/1_000_000))))

	tp.Print()

}

type tablePrinter struct {
	w         *tabwriter.Writer
	maxLength int
	title     string
}

type printItem struct {
	val        string
	alignRight bool
}

func item(val string, alignRightOpt ...bool) printItem {
	alignRight := false
	if len(alignRightOpt) > 0 {
		alignRight = alignRightOpt[0]
	}
	return printItem{
		val:        val,
		alignRight: alignRight,
	}
}

func newTablePrinter() *tablePrinter {
	w := tabwriter.NewWriter(
		os.Stdout, //output
		0,         // min width
		1,         // tabwidth
		2,         // padding
		' ',       // padding character
		0,         // align left
	)

	return &tablePrinter{
		w:         w,
		maxLength: 0,
	}
}

func (tp *tablePrinter) AddRecord(items ...string) {
	tp.printRecord(items...)
}

func (tp *tablePrinter) AlignRight() {
	tp.w.Init(
		os.Stdout, //output
		0,         // min width
		1,         // tabwidth
		2,         // padding
		' ',       // padding character
		tabwriter.AlignRight,
	) // flags
}

func (tp *tablePrinter) AlignLeft() {
	tp.w.Init(
		os.Stdout, //output
		0,         // min width
		1,         // tabwidth
		2,         // padding
		' ',       // padding character
		0,         // align left
	) // flags
}

func (tp *tablePrinter) printRecord(rec ...string) {
	var val string
	for i, item := range rec {
		switch i {
		case 0:
			val = item
		case len(rec) - 1:
			val += fmt.Sprintf("\t%s\n", item)
		default:
			val += fmt.Sprintf("\t%s", item)
		}
	}

	nbytes, err := tp.w.Write([]byte(val))
	if err != nil {
		panic(err)
	}

	if nbytes > tp.maxLength {
		tp.maxLength = nbytes
	}
}

func (tp *tablePrinter) Print() {
	printBorder(tp.maxLength)
	printLine(tp.maxLength, tp.title)
	printBorder(tp.maxLength)
	tp.w.Flush()
	printBorder(tp.maxLength)
}

func padRight(val interface{}, rightEnd int) string {
	value := fmt.Sprintf("%v", val)
	pad := fmt.Sprintf("%s", strings.Repeat(" ", rightEnd-len(value)))
	return fmt.Sprintf("%s%s", pad, value)
}

func printLine(lineLength int, value string) {
	fmt.Printf("| %s %s\n", value, padRight("|", lineLength-len(value)-1))
}

func printBorder(length int) {
	line := fmt.Sprintf("%s", strings.Repeat("-", length))
	fmt.Printf("+%s+\n", line)
}
