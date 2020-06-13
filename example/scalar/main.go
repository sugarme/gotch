package main

import (
	"fmt"
	"log"

	"github.com/sugarme/gotch/wrapper"
)

func main() {

	s := wrapper.FloatScalar(float64(1.23))
	fmt.Printf("scalar value: %v\n", s)

	intVal, err := s.ToInt()
	if err != nil {
		panic(err)
	}
	floatVal, err := s.ToFloat()
	if err != nil {
		panic(err)
	}
	strVal, err := s.ToString()
	if err != nil {
		panic(err)
	}

	fmt.Printf("scalar to int64 value: %v\n", intVal)
	fmt.Printf("scalar to float64 value: %v\n", floatVal)
	fmt.Printf("scalar to string value: %v\n", strVal)

	s.Drop() // will set scalar to zero
	fmt.Printf("scalar value: %v\n", s)

	zeroVal, err := s.ToInt()
	if err != nil {
		log.Fatalf("Panic: %v\n", err)
	}

	fmt.Printf("Won't expect this val: %v\n", zeroVal)
}
