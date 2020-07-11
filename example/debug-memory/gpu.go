package main

import (
	"fmt"
	"log"
	"os/exec"
)

func GPUInfo() {
	// Print out GPU used
	nvidia := "nvidia-smi"
	cmd := exec.Command(nvidia)
	stdout, err := cmd.Output()

	if err != nil {
		log.Fatal(err.Error())
	}

	fmt.Println(string(stdout))
}
