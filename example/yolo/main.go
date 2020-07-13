package main

import (
	"fmt"
	// "flag"
	"log"
	"path/filepath"
)

const configName = "yolo-v3.cfg"

func init() {

}

func main() {

	configPath, err := filepath.Abs(configName)
	if err != nil {
		log.Fatal(err)
	}

	var darknet Darknet = ParseConfig(configPath)

	fmt.Printf("darknet number of parameters: %v\n", len(darknet.parameters))
	fmt.Printf("darknet number of blocks: %v\n", len(darknet.blocks))
}
