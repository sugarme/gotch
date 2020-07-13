package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/sugarme/gotch/nn"
	ts "github.com/sugarme/gotch/tensor"
)

type block struct {
	blockType  *string // optional
	parameters map[string]string
}

func (b block) get(key string) (retVal string) {
	val, ok := b.parameters[key]
	if !ok {
		log.Fatalf("Cannot find %v in net parameters.\n", key)
	}

	return val
}

type Darknet struct {
	blocks     []block
	parameters map[string]string
}

func (d Darknet) get(key string) (retVal string) {
	val, ok := d.parameters[key]
	if !ok {
		log.Fatalf("Cannot find %v in net parameters.\n", key)
	}

	return val
}

type accumulator struct {
	parameters map[string]string
	net        Darknet
	blockType  *string // optional
}

func newAccumulator() (retVal accumulator) {

	return accumulator{
		blockType:  nil,
		parameters: make(map[string]string, 0),
		net: Darknet{
			blocks:     make([]block, 0),
			parameters: make(map[string]string, 0),
		},
	}
}

func (acc *accumulator) finishBlock() {
	if acc.blockType != nil {
		if *acc.blockType == "net" {
			acc.net.parameters = acc.parameters
		} else {
			block := block{
				blockType:  acc.blockType,
				parameters: acc.parameters,
			}
			acc.net.blocks = append(acc.net.blocks, block)
		}

		// clear parameters
		acc.parameters = make(map[string]string, 0)
	}

	acc.blockType = nil
}

func ParseConfig(path string) (retVal Darknet) {

	acc := newAccumulator()

	var lines []string

	// Read file line by line
	// Ref. https://stackoverflow.com/questions/8757389
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		lines = append(lines, line)
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	for _, line := range lines {

		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimSpace(line)

		if strings.HasPrefix(line, "[") {
			// make sure line ends with "]"
			if !strings.HasSuffix(line, "]") {
				log.Fatalf("Line doesn't end with ']'\n")
			}

			line = strings.TrimPrefix(line, "[")
			line = strings.TrimSuffix(line, "]")

			acc.finishBlock()
			acc.blockType = &line
		} else {
			var keyValue []string
			keyValue = strings.Split(line, "=")
			if len(keyValue) != 2 {
				log.Fatalf("Missing equal for line: %v\n", line)
			}

			// // Ensure key does not exist
			// if _, ok := acc.parameters[keyValue[0]]; ok {
			// log.Fatalf("Multiple values for key - %v\n", line)
			// }

			acc.parameters[keyValue[0]] = keyValue[1]

		}
	} // end of for

	acc.finishBlock()

	return acc.net
}

type (
	Layer    = ts.ModuleT
	Route    = []uint
	Shortcut = uint
	Yolo     struct {
		V1 int64
		V2 []int64
	}
)

func conv(vs nn.Path, index uint, p int64, b block) (retVal1 int64, retVal2 interface{}) {

	activation := b.get("activation")

	filters, err := strconv.ParseInt(b.get("filters"), 10, 64)
	if err != nil {
		log.Fatal(err)
	}

	pad, err := strconv.ParseInt(b.get("pad"), 10, 64)
	if err != nil {
		log.Fatal(err)
	}

	size, err := strconv.ParseInt(b.get("size"), 10, 64)
	if err != nil {
		log.Fatal(err)
	}

	stride, err := strconv.ParseInt(b.get("stride"), 10, 64)
	if err != nil {
		log.Fatal(err)
	}

	if pad != 0 {
		pad = (size - 1) / 2
	} else {
		pad = 0
	}

	var (
		bn   *nn.BatchNorm
		bias bool
	)
	if pStr, ok := b.parameters["batch_normalize"]; ok {
		p, err := strconv.ParseInt(pStr, 10, 64)
		if err != nil {
			log.Fatal(err)
		}

		if p != 0 {
			sub := vs.Sub(fmt.Sprintf("batch_norm_%v", index))
			bnVal := nn.BatchNorm2D(sub, filters, nn.DefaultBatchNormConfig())
			bn = &bnVal
			bias = false
		}
	} else {
		bn = nil
		bias = true
	}

	convConfig := nn.DefaultConv2DConfig()
	convConfig.Stride = []int64{stride, stride}
	convConfig.Padding = []int64{pad, pad}
	convConfig.Bias = bias

	conv := nn.NewConv2D(vs.Sub(fmt.Sprintf("conv_%v", index)), p, filters, size, convConfig)

	var leaky bool
	switch activation {
	case "leaky":
		leaky = true
	case "linear":
		leaky = false
	default:
		log.Fatalf("Unsupported activation(%v)\n", activation)
	}

	fn := nn.NewFuncT(func(xs ts.Tensor, train bool) (res ts.Tensor) {
		tmp1 := xs.Apply(conv)

		var tmp2 ts.Tensor

		if bn != nil {
			tmp2 = tmp1.ApplyT(*bn, train)
			tmp1.MustDrop()
		} else {
			tmp2 = tmp1
		}

		if leaky {
			tmp2Mul := tmp2.MustMul1(ts.FloatScalar(0.1), false)
			res = tmp2.MustMax1(tmp2Mul)
			tmp2Mul.MustDrop()
			tmp2.MustDrop()
		} else {
			res = tmp2
		}

		return res
	})

	return filters, fn
}
