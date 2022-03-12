package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"reflect"
	"strconv"
	"strings"

	"github.com/sugarme/gotch"
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/gotch/ts"
)

type Block struct {
	BlockType  *string // optional
	Parameters map[string]string
}

func (b *Block) get(key string) string {
	val, ok := b.Parameters[key]
	if !ok {
		log.Fatalf("Cannot find %v in Block parameters.\n", key)
	}

	return val
}

type Darknet struct {
	Blocks     []Block
	Parameters map[string]string
}

func (d *Darknet) get(key string) string {
	val, ok := d.Parameters[key]
	if !ok {
		log.Fatalf("Cannot find %v in Darknet parameters.\n", key)
	}

	return val
}

type Accumulator struct {
	Parameters map[string]string
	Net        *Darknet
	BlockType  *string // optional
}

func newAccumulator() *Accumulator {

	return &Accumulator{
		BlockType:  nil,
		Parameters: make(map[string]string, 0),
		Net: &Darknet{
			Blocks:     make([]Block, 0),
			Parameters: make(map[string]string, 0),
		},
	}
}

func (acc *Accumulator) finishBlock() {
	if acc.BlockType != nil {
		if *acc.BlockType == "net" {
			acc.Net.Parameters = acc.Parameters
		} else {
			block := Block{
				BlockType:  acc.BlockType,
				Parameters: acc.Parameters,
			}
			acc.Net.Blocks = append(acc.Net.Blocks, block)
		}

		// clear parameters
		acc.Parameters = make(map[string]string, 0)
	}

	acc.BlockType = nil
}

func ParseConfig(path string) *Darknet {

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

	for _, ln := range lines {
		line := ln
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimSpace(line)           // trim all spaces before and after
		line = strings.ReplaceAll(line, " ", "") // trim all space in between
		if strings.HasPrefix(line, "[") {
			// make sure line ends with "]"
			if !strings.HasSuffix(line, "]") {
				log.Fatalf("Line doesn't end with ']'\n")
			}
			line = strings.TrimPrefix(line, "[")
			line = strings.TrimSuffix(line, "]")

			acc.finishBlock()
			acc.BlockType = &line

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

			acc.Parameters[keyValue[0]] = keyValue[1]
		}
	} // end of for

	acc.finishBlock()

	return acc.Net
}

type (
	Layer struct {
		Val nn.FuncT
	}
	Route struct {
		TsIdxs []uint
	}
	Shortcut struct {
		TsIdx uint // tensor index
	}

	Anchor []int64

	Yolo struct {
		Classes int64
		Anchors []Anchor
	}

	ChannelsBl struct {
		Channels int64
		Bl       interface{}
	}
)

func conv(vs *nn.Path, index uint, p int64, b *Block) (retVal1 int64, retVal2 interface{}) {

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
	if pStr, ok := b.Parameters["batch_normalize"]; ok {
		p, err := strconv.ParseInt(pStr, 10, 64)
		if err != nil {
			log.Fatal(err)
		}

		if p != 0 {
			sub := vs.Sub(fmt.Sprintf("batch_norm_%v", index))
			bnVal := nn.BatchNorm2D(sub, filters, nn.DefaultBatchNormConfig())
			bn = bnVal
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

	fn := nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		tmp1 := xs.Apply(conv)

		var tmp2 *ts.Tensor

		if bn != nil {
			tmp2 = tmp1.ApplyT(bn, train)
			tmp1.MustDrop()
		} else {
			tmp2 = tmp1
		}

		var res *ts.Tensor
		if leaky {
			tmp2Mul := tmp2.MustMulScalar(ts.FloatScalar(0.1), false)
			res = tmp2.MustMaximum(tmp2Mul, true)
			tmp2Mul.MustDrop()
		} else {
			res = tmp2
		}

		return res
	})

	return filters, Layer{Val: fn}
}

func upsample(prevChannels int64) (retVal1 int64, retVal2 interface{}) {
	layer := nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {
		// []int64{n, c, h, w}
		res, err := xs.Size4()
		if err != nil {
			log.Fatal(err)
		}
		h := res[2]
		w := res[3]

		return xs.MustUpsampleNearest2d([]int64{h * 2, w * 2}, []float64{2.0}, []float64{2.0}, false)
	})

	return prevChannels, Layer{Val: layer}
}

func intListOfString(s string) []int64 {
	var retVal []int64
	strs := strings.Split(s, ",")
	for _, str := range strs {
		str = strings.TrimSpace(str)
		i, err := strconv.ParseInt(str, 10, 64)
		if err != nil {
			log.Fatal(err)
		}
		retVal = append(retVal, i)
	}

	return retVal
}

func uintOfIndex(index uint, i int64) uint {
	if i >= 0 {
		return uint(i)
	} else {
		return uint(int64(index) + i)
	}
}

func route(index uint, p []ChannelsBl, blk *Block) (retVal1 int64, retVal2 interface{}) {
	intLayers := intListOfString(blk.get("layers"))

	var layers []uint
	for _, l := range intLayers {
		layers = append(layers, uintOfIndex(index, l))
	}

	var channels int64
	for _, l := range layers {
		channels += p[l].Channels
	}

	return channels, Route{TsIdxs: layers}
}

func shortcut(index uint, p int64, blk *Block) (retVal1 int64, retVal2 interface{}) {
	fromStr := blk.get("from")

	from, err := strconv.ParseInt(fromStr, 10, 64)
	if err != nil {
		log.Fatal(err)
	}

	return p, Shortcut{TsIdx: uintOfIndex(index, from)}
}

func yolo(p int64, blk *Block) (retVal1 int64, retVal2 interface{}) {
	classesStr := blk.get("classes")
	classes, err := strconv.ParseInt(classesStr, 10, 64)
	if err != nil {
		log.Fatal(err)
	}

	// anchorsStr := blk.get("anchors")
	flat := intListOfString(blk.get("anchors"))

	if (len(flat) % 2) != 0 {
		log.Fatalf("Expected even number of flat")
	}

	var anchors [][]int64

	for i := 0; i < len(flat)/2; i++ {
		anchors = append(anchors, []int64{flat[2*i], flat[2*i+1]})
	}

	intMask := intListOfString(blk.get("mask"))

	var retAnchors []Anchor
	for _, i := range intMask {
		retAnchors = append(retAnchors, anchors[i])
	}

	return p, Yolo{Classes: classes, Anchors: retAnchors}
}

// Apply f to a slice of tensor xs and replace xs values with f output.
func sliceApplyAndSet(xs *ts.Tensor, start int64, len int64, f func(*ts.Tensor) *ts.Tensor) {
	slice := xs.MustNarrow(2, start, len, false)
	src := f(slice)

	slice.Copy_(src)
	src.MustDrop()
	slice.MustDrop()
}

func detect(xs *ts.Tensor, imageHeight int64, classes int64, anchors []Anchor) *ts.Tensor {

	device, err := xs.Device()

	size4, err := xs.Size4()
	if err != nil {
		log.Fatal(err)
	}
	bsize := size4[0]
	height := size4[2]

	stride := imageHeight / height
	gridSize := imageHeight / stride
	bboxAttrs := classes + 5
	nanchors := int64(len(anchors))

	tmp1 := xs.MustView([]int64{bsize, bboxAttrs * nanchors, gridSize * gridSize}, false)
	tmp2 := tmp1.MustTranspose(1, 2, true)
	tmp3 := tmp2.MustContiguous(true)
	xsTs := tmp3.MustView([]int64{bsize, gridSize * gridSize * nanchors, bboxAttrs}, true)

	if err != nil {
		log.Fatal(err)
	}
	grid := ts.MustArange(ts.IntScalar(gridSize), gotch.Float, device)
	a := grid.MustRepeat([]int64{gridSize, 1}, true)
	bTmp := a.MustT(false)
	b := bTmp.MustContiguous(true)

	xOffset := a.MustView([]int64{-1, 1}, true)
	yOffset := b.MustView([]int64{-1, 1}, true)
	xyOffsetTmp1 := ts.MustCat([]ts.Tensor{*xOffset, *yOffset}, 1)
	xyOffsetTmp2 := xyOffsetTmp1.MustRepeat([]int64{1, nanchors}, true)
	xyOffsetTmp3 := xyOffsetTmp2.MustView([]int64{-1, 2}, true)
	xyOffset := xyOffsetTmp3.MustUnsqueeze(0, true)

	var flatAnchors []int64
	for _, a := range anchors {
		flatAnchors = append(flatAnchors, a...)
	}

	var anchorVals []float32
	for _, a := range flatAnchors {
		v := float32(a) / float32(stride)
		anchorVals = append(anchorVals, v)
	}

	anchorsTmp1 := ts.MustOfSlice(anchorVals)
	anchorsTmp2 := anchorsTmp1.MustView([]int64{-1, 2}, true)
	anchorsTmp3 := anchorsTmp2.MustRepeat([]int64{gridSize * gridSize, 1}, true)
	anchorsTs := anchorsTmp3.MustUnsqueeze(0, true).MustTo(device, true)

	sliceApplyAndSet(xsTs, 0, 2, func(xs *ts.Tensor) *ts.Tensor {
		tmp := xs.MustSigmoid(false)
		return tmp.MustAdd(xyOffset, true)
	})

	sliceApplyAndSet(xsTs, 4, classes+1, func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustSigmoid(false)
	})

	sliceApplyAndSet(xsTs, 2, 2, func(xs *ts.Tensor) *ts.Tensor {
		tmp := xs.MustExp(false)
		return tmp.MustMul(anchorsTs, true)
	})

	sliceApplyAndSet(xsTs, 0, 4, func(xs *ts.Tensor) *ts.Tensor {
		return xs.MustMulScalar(ts.IntScalar(stride), false)
	})

	// TODO: delete all middle tensors.
	return xsTs
}

func (dn *Darknet) Height() int64 {
	imageHeightStr := dn.get("height")
	retVal, err := strconv.ParseInt(imageHeightStr, 10, 64)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (dn *Darknet) Width() int64 {
	imageWidthStr := dn.get("width")
	retVal, err := strconv.ParseInt(imageWidthStr, 10, 64)
	if err != nil {
		log.Fatal(err)
	}

	return retVal
}

func (dn *Darknet) BuildModel(vs *nn.Path) nn.FuncT {
	var blocks []ChannelsBl // Param is a struct{int64, interface{}}
	var prevChannels int64 = 3

	for index, blk := range dn.Blocks {
		var channels int64
		var bl interface{}

		switch *blk.BlockType {
		case "convolutional":
			channels, bl = conv(vs.Sub(fmt.Sprintf("%v", index)), uint(index), prevChannels, &blk)
		case "upsample":
			channels, bl = upsample(prevChannels)
		case "shortcut":
			channels, bl = shortcut(uint(index), prevChannels, &blk)
		case "route":
			channels, bl = route(uint(index), blocks, &blk)
		case "yolo":
			channels, bl = yolo(prevChannels, &blk)
		default:
			log.Fatalf("Unsupported block type: %v\n", *blk.BlockType)
		}
		prevChannels = channels
		blocks = append(blocks, ChannelsBl{channels, bl})
	}

	imageHeight := dn.Height()

	retVal := nn.NewFuncT(func(xs *ts.Tensor, train bool) *ts.Tensor {

		var prevYs []ts.Tensor = make([]ts.Tensor, 0)
		var detections []ts.Tensor = make([]ts.Tensor, 0)

		// NOTE: we will delete all tensors in prevYs after looping
		for _, b := range blocks {
			blkTyp := reflect.TypeOf(b.Bl)
			var ysTs *ts.Tensor
			switch blkTyp.Name() {
			case "Layer":
				layer := b.Bl.(Layer)
				xsTs := xs
				if len(prevYs) > 0 {
					xsTs = &prevYs[len(prevYs)-1] // last prevYs element
				}
				ysTs = layer.Val.ForwardT(xsTs, train)
			case "Route":
				route := b.Bl.(Route)
				var layers []ts.Tensor
				for _, i := range route.TsIdxs {
					layers = append(layers, prevYs[int(i)])
				}
				ysTs = ts.MustCat(layers, 1)

			case "Shortcut":
				from := b.Bl.(Shortcut).TsIdx
				addTs := &prevYs[int(from)]
				last := prevYs[len(prevYs)-1]
				ysTs = last.MustAdd(addTs, false)
			case "Yolo":
				classes := b.Bl.(Yolo).Classes
				anchors := b.Bl.(Yolo).Anchors
				xsTs := xs
				if len(prevYs) > 0 {
					xsTs = &prevYs[len(prevYs)-1]
				}

				dt := detect(xsTs, imageHeight, classes, anchors)

				detections = append(detections, *dt)

				ysTs = ts.NewTensor()

			default:
				// log.Fatalf("BuildModel - FuncT - Unsupported block type: %v\n", blkTyp.Name())
			} // end of Switch

			prevYs = append(prevYs, *ysTs)
		} // end of For loop

		res := ts.MustCat(detections, 1)

		// Now, free-up memory held up by prevYs
		for _, t := range prevYs {
			if t.MustDefined() {
				// fmt.Printf("will delete ts: %v\n", t)
				// NOTE: if t memory is delete previously (in switch-case), there will be panic!
				t.MustDrop()
			}
		}

		return res
	}) // end of NewFuncT

	return retVal
}
