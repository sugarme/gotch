package main

import (
	"image"
	"image/color"

	// "image/jpeg"
	"io/ioutil"

	"flag"
	"log"
	"os"
	"path/filepath"

	"golang.org/x/image/draw"
	"golang.org/x/image/font"

	"github.com/sugarme/gotch/example/yolo/freetype"
	"github.com/sugarme/gotch/ts"
)

var (
	dpi      = flag.Float64("dpi", 72, "screen resolution in Dots Per Inch")
	fontfile = flag.String("fontfile", "luxisr.ttf", "filename of the ttf font")
	hinting  = flag.String("hinting", "none", "none | full")
	size     = flag.Float64("size", 12, "font size in points")
	spacing  = flag.Float64("spacing", 1.2, "line spacing (e.g. 2 means double spaced)")
	wonb     = flag.Bool("whiteonblack", false, "white text on a black background")
	bound    = flag.Bool("bound", true, "generates image with minimum size for the text")
)

func loadImage(file string) (retVal image.Image, err error) {
	imagePath, err := filepath.Abs(file)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(imagePath)
	if err != nil {
		return nil, err
	}

	img, _, err := image.Decode(f)
	return img, err
}

func textToImageTs(text []string) *ts.Tensor {
	offset := 0

	flag.Parse()

	// Read font data
	fontBytes, err := ioutil.ReadFile(*fontfile)
	if err != nil {
		log.Println(err)
		return nil
	}

	f, err := freetype.ParseFont(fontBytes)
	if err != nil {
		log.Println(err)
		return nil
	}

	var width, height int
	// Initialize the context.
	c := freetype.NewContext()
	c.SetDPI(*dpi)
	c.SetFont(f)
	c.SetFontSize(*size)

	switch *hinting {
	default:
		c.SetHinting(font.HintingNone)
	case "full":
		c.SetHinting(font.HintingFull)
	}

	// Measure the text to calculate the minimum size of the image
	if *bound {
		pt := freetype.Pt(offset, offset+int(c.PointToFixed(*size)>>6))
		for _, s := range text {
			ptr, err := c.MeasureString(s, pt)
			if err != nil {
				log.Println(err)
				return nil
			}
			pt.Y += c.PointToFixed(*size * *spacing)
			x := int(ptr.X >> 6)
			if x > width {
				width = x
			}
		}
		width += offset
		height = int(pt.Y)>>6 - int(c.PointToFixed(*size)>>6)
		// Use default size for the image
	} else {
		width = 640
		height = 480
	}

	// Creates image with the specified size
	fg, bg := image.Black, image.White
	ruler := color.RGBA{0xdd, 0xdd, 0xdd, 0xff}
	if *wonb {
		fg, bg = image.White, image.Black
		ruler = color.RGBA{0x22, 0x22, 0x22, 0xff}
	}
	rgba := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.Draw(rgba, rgba.Bounds(), bg, image.ZP, draw.Src)
	c.SetClip(rgba.Bounds())
	c.SetDst(rgba)
	c.SetSrc(fg)

	// Draw the guidelines
	for i := 0; i < 200; i++ {
		rgba.Set(offset, offset+i, ruler)
		rgba.Set(offset+i, offset, ruler)
	}

	// Draw the text.
	pt := freetype.Pt(offset, offset+int(c.PointToFixed(*size)>>6))
	for _, s := range text {
		_, err = c.DrawString(s, pt)
		if err != nil {
			log.Println(err)
			return nil
		}
		pt.Y += c.PointToFixed(*size * *spacing)
	}

	var rgb []float64
	var r, g, b []float64
	for i := 0; i < len(rgba.Pix); i += 4 {
		start := i
		r = append(r, float64(rgba.Pix[start])/255.0)
		g = append(g, float64(rgba.Pix[start+1])/255.0)
		b = append(b, float64(rgba.Pix[start+2])/255.0)
	}

	rgb = append(rgb, r...)
	rgb = append(rgb, g...)
	rgb = append(rgb, b...)

	w := int64(rgba.Rect.Dx())
	h := int64(rgba.Rect.Dy())

	return ts.MustOfSlice(rgb).MustView([]int64{3, h, w}, false)
}
