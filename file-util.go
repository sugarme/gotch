package gotch

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path"
	"strconv"
	"strings"
)

// This file provides functions to work with local dataset cache, ...

// ModelUrls maps model name to its pretrained URL.
//
// This URLS taken from separate models in pytorch/vision repository
// https://github.com/pytorch/vision/tree/main/torchvision/models
var ModelUrls map[string]string = map[string]string{
	"alexnet": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",

	"convnext_tiny":  "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
	"convnext_small": "https://download.pytorch.org/models/convnext_small-0c510722.pth",
	"convnext_base":  "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
	"convnext_large": "https://download.pytorch.org/models/convnext_large-ea097f82.pth",

	"densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
	"densenet169": "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
	"densenet201": "https://download.pytorch.org/models/densenet201-c1103571.pth",
	"densenet161": "https://download.pytorch.org/models/densenet161-8d451a50.pth",

	//Weights ported from https://github.com/rwightman/pytorch-image-models/
	"efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
	"efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
	"efficientnet_b2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
	"efficientnet_b3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
	"efficientnet_b4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
	//Weights ported from https://github.com/lukemelas/EfficientNet-PyTorch/
	"efficientnet_b5": "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
	"efficientnet_b6": "https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
	"efficientnet_b7": "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",

	//GoogLeNet ported from TensorFlow
	"googlenet": "https://download.pytorch.org/models/googlenet-1378be20.pth",

	//Inception v3 ported from TensorFlow
	"inception_v3_google": "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth",

	"mnasnet0_5":  "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
	"mnasnet0_75": "",
	"mnasnet1_0":  "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
	"mnasnet1_3":  "",

	"mobilenet_v2":       "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
	"mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
	"mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",

	"regnet_y_400mf": "https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth",
	"regnet_y_800mf": "https://download.pytorch.org/models/regnet_y_800mf-1b27b58c.pth",
	"regnet_y_1_6gf": "https://download.pytorch.org/models/regnet_y_1_6gf-b11a554e.pth",
	"regnet_y_3_2gf": "https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth",
	"regnet_y_8gf":   "https://download.pytorch.org/models/regnet_y_8gf-d0d0e4a8.pth",
	"regnet_y_16gf":  "https://download.pytorch.org/models/regnet_y_16gf-9e6ed7dd.pth",
	"regnet_y_32gf":  "https://download.pytorch.org/models/regnet_y_32gf-4dee3f7a.pth",
	"regnet_x_400mf": "https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth",
	"regnet_x_800mf": "https://download.pytorch.org/models/regnet_x_800mf-ad17e45c.pth",
	"regnet_x_1_6gf": "https://download.pytorch.org/models/regnet_x_1_6gf-e3633e7f.pth",
	"regnet_x_3_2gf": "https://download.pytorch.org/models/regnet_x_3_2gf-f342aeae.pth",
	"regnet_x_8gf":   "https://download.pytorch.org/models/regnet_x_8gf-03ceed89.pth",
	"regnet_x_16gf":  "https://download.pytorch.org/models/regnet_x_16gf-2007eb11.pth",
	"regnet_x_32gf":  "https://download.pytorch.org/models/regnet_x_32gf-9d47f8d0.pth",

	"resnet18":         "https://download.pytorch.org/models/resnet18-f37072fd.pth",
	"resnet34":         "https://download.pytorch.org/models/resnet34-b627a593.pth",
	"resnet50":         "https://download.pytorch.org/models/resnet50-0676ba61.pth",
	"resnet101":        "https://download.pytorch.org/models/resnet101-63fe2227.pth",
	"resnet152":        "https://download.pytorch.org/models/resnet152-394f9c45.pth",
	"resnext50_32x4d":  "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
	"resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
	"wide_resnet50_2":  "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
	"wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",

	"shufflenetv2_x0.5": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
	"shufflenetv2_x1.0": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
	"shufflenetv2_x1.5": "",
	"shufflenetv2_x2.0": "",

	"squeezenet1_0": "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
	"squeezenet1_1": "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",

	"vgg11":    "https://download.pytorch.org/models/vgg11-8a719046.pth",
	"vgg13":    "https://download.pytorch.org/models/vgg13-19584684.pth",
	"vgg16":    "https://download.pytorch.org/models/vgg16-397923af.pth",
	"vgg19":    "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
	"vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
	"vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
	"vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
	"vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",

	"vit_b_16": "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
	"vit_b_32": "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
	"vit_l_16": "https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
	"vit_l_32": "https://download.pytorch.org/models/vit_l_32-c7638314.pth",
}

// CachedPath resolves and caches data based on input string, then returns fullpath to the cached data.
//
// Parameters:
// - `filenameOrUrl`: full path to filename or url
//
// CachedPath does several things consequently:
// 1. Resolves input string to a  fullpath cached filename candidate.
// 2. Check it at `CachedDir`, if exists, then return the candidate. If not
// 3. Retrieves and Caches data to `CachedDir` and returns path to cached data
func CachedPath(filenameOrUrl string, folderOpt ...string) (resolvedPath string, err error) {
	filename := path.Base(filenameOrUrl)
	// Resolves to "candidate" filename at `CachedDir`
	fullPath := CachedDir
	if len(folderOpt) > 0 {
		fullPath = fmt.Sprintf("%v/%v", CachedDir, folderOpt[0])
	}

	cachedFileCandidate := fmt.Sprintf("%s/%s", fullPath, filename)

	// 1. Cached candidate file exists
	if _, err := os.Stat(cachedFileCandidate); err == nil {
		return cachedFileCandidate, nil
	}

	// 2. If valid fullpath to local file, caches it and return cached filename
	if _, err := os.Stat(filenameOrUrl); err == nil {
		err := copyFile(filenameOrUrl, cachedFileCandidate)
		if err != nil {
			return "", err
		}
		return cachedFileCandidate, nil
	}

	// 3. Cached candidate file NOT exist. Try to download it and save to `CacheDir`
	if isValidURL(filenameOrUrl) {
		if _, err := http.Get(filenameOrUrl); err == nil {
			err := downloadFile(filenameOrUrl, cachedFileCandidate)
			if err != nil {
				return "", err
			}

			return cachedFileCandidate, nil
		} else {
			fmt.Printf("Error: %v\n", err)
			err = fmt.Errorf("Unable to parse %q as a URL or as a local path.\n", filenameOrUrl)
			return "", err
		}
	}

	// Not resolves
	err = fmt.Errorf("Unable to parse %q as a URL or as a local path.\n", filenameOrUrl)
	return "", err
}

func isValidURL(url string) bool {

	// TODO: implement
	return true
}

// downloadFile downloads file from URL and stores it in local filepath.
// It writes to the destination file as it downloads it, without loading
// the entire file into memory. An `io.TeeReader` is passed into Copy()
// to report progress on the download.
func downloadFile(url string, filepath string) error {
	// Create path if not existing
	dir := path.Dir(filepath)
	filename := path.Base(filepath)
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		if err := os.MkdirAll(dir, 0755); err != nil {
			log.Fatal(err)
		}
	}

	// Create the file with .tmp extension, so that we won't overwrite a
	// file until it's downloaded fully
	out, err := os.Create(filepath + ".tmp")
	if err != nil {
		return err
	}
	defer out.Close()

	// Get the data
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Check server response
	if resp.StatusCode != http.StatusOK {
		err := fmt.Errorf("bad status: %s(%v)", resp.Status, resp.StatusCode)
		if resp.StatusCode == 404 {
			err = fmt.Errorf("download file not found: %q for downloading", url)
		} else {
			err = fmt.Errorf("download file failed: %q", url)
		}
		return err
	}

	// the total file size to download
	size, _ := strconv.Atoi(resp.Header.Get("Content-Length"))
	downloadSize := uint64(size)

	// Create our bytes counter and pass it to be used alongside our writer
	counter := &writeCounter{FileSize: downloadSize}
	_, err = io.Copy(out, io.TeeReader(resp.Body, counter))
	if err != nil {
		return err
	}

	fmt.Printf("\r%s... %s/%s completed", filename, byteCountIEC(counter.Total), byteCountIEC(counter.FileSize))
	// The progress use the same line so print a new line once it's finished downloading
	fmt.Println()

	// Rename the tmp file back to the original file
	err = os.Rename(filepath+".tmp", filepath)
	if err != nil {
		return err
	}

	return nil
}

// writeCounter counts the number of bytes written to it. By implementing the Write method,
// it is of the io.Writer interface and we can pass this into io.TeeReader()
// Every write to this writer, will print the progress of the file write.
type writeCounter struct {
	Total    uint64
	FileSize uint64
}

func (wc *writeCounter) Write(p []byte) (int, error) {
	n := len(p)
	wc.Total += uint64(n)
	wc.printProgress()
	return n, nil
}

// PrintProgress prints the progress of a file write
func (wc writeCounter) printProgress() {
	// Clear the line by using a character return to go back to the start and remove
	// the remaining characters by filling it with spaces
	fmt.Printf("\r%s", strings.Repeat(" ", 50))

	// Return again and print current status of download
	fmt.Printf("\rDownloading... %s/%s", byteCountIEC(wc.Total), byteCountIEC(wc.FileSize))
}

// byteCountIEC converts bytes to human-readable string in binary (IEC) format.
func byteCountIEC(b uint64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := uint64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB",
		float64(b)/float64(div), "KMGTPE"[exp])
}

func copyFile(src, dst string) error {
	sourceFileStat, err := os.Stat(src)
	if err != nil {
		return err
	}

	if !sourceFileStat.Mode().IsRegular() {
		return fmt.Errorf("%s is not a regular file", src)
	}

	source, err := os.Open(src)
	if err != nil {
		return err
	}
	defer source.Close()

	destination, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destination.Close()
	_, err = io.Copy(destination, source)
	return err
}

// CleanCache removes all files cached at `CachedDir`
func CleanCache() error {
	err := os.RemoveAll(CachedDir)
	if err != nil {
		err = fmt.Errorf("CleanCache() failed: %w", err)
		return err
	}

	return nil
}
