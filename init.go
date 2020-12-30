package gotch

import (
	// "flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/user"
	"path"
	"strings"
)

var (
	GotchCacheDir   string
	TorchDir        string
	IsCudaAvailable bool = false
	HomeDir         string
)

func init() {
	if v, ok := os.LookupEnv("TORCH_VERSION"); ok {
		TorchVersion = v
	}
	if v, ok := os.LookupEnv("TORCH_DIR"); ok {
		TorchDir = v
	}

	IsCudaAvailable = isCudaAvailable()
	if !isCudaAvailable() {
		CudaVersion = "cpu"
	}

	HomeDir = homeDir()
	GotchCacheDir = fmt.Sprintf("%v/.cache/gotch", HomeDir)

	InstallLibtorch()
}

// isCudaAvailable check whether cuda is installed using 'nvcc'
func isCudaAvailable() bool {
	_, err := exec.Command("nvcc", "--version").Output()
	if err != nil {
		log.Printf("CUDA is not detected using 'nvcc --version'\n")
		return false
	}
	return true
}

// homeDir returns home directory of current user.
func homeDir() string {
	usr, err := user.Current()
	if err != nil {
		log.Fatal(err)
	}

	return usr.HomeDir
}

func InstallLibtorch() {
	dev := strings.ReplaceAll(CudaVersion, ".", "")
	cu := "cpu"
	if !strings.Contains(dev, "cpu") {
		cu = fmt.Sprintf("cu%v", dev)
	}
	url := fmt.Sprintf("https://download.pytorch.org/libtorch/%v/libtorch-cxx11-abi-shared-with-deps-%v%%2B%v.zip", cu, TorchVersion, cu)

	// Create dir if not exist
	TorchDir = fmt.Sprintf("%v/libtorch-%v-%v", GotchCacheDir, TorchVersion, cu)
	if _, err := os.Stat(TorchDir); os.IsNotExist(err) {
		os.MkdirAll(TorchDir, 0755)
	}
	// install libtorch if not done yet.
	installLibtorch(url)

	// Export to current context so that shell script can catch them
	libtorch := fmt.Sprintf("%v/libtorch", TorchDir)
	os.Setenv("GOTCH_LIBTORCH", libtorch)
	os.Setenv("GOTCH_CUDA_VERSION", CudaVersion)
	os.Setenv("GOTCH_VER", GotchVersion)

	envFile := fmt.Sprintf("%v/env.sh", CurrDir())
	_, err := exec.Command(envFile).Output()
	if err != nil {
		log.Fatal(err)
	}
}

func installLibtorch(url string) {
	filename := path.Base(url)
	cachedFileCandidate := fmt.Sprintf("%s/%v", TorchDir, filename)

	// Check whether zip file exists otherwise, download it
	if _, err := os.Stat(cachedFileCandidate); err == nil {
		// zip file exists
		// check one file 'libc10.so' if exists, assuming libtorch has been installed.
		libc10 := fmt.Sprintf("%v/libtorch/lib/libc10.so", TorchDir)
		if _, err := os.Stat(libc10); err == nil {
			return
		}
	} else {
		// download zip file
		err := downloadFile(url, cachedFileCandidate)
		if err != nil {
			log.Fatal(err)
		}
	}

	// Unzip
	log.Printf("Unzipping...")
	err := Unzip(cachedFileCandidate, TorchDir)
	if err != nil {
		log.Fatal(err)
	}
}

func Hello() {
	fmt.Println("Hello")
}
