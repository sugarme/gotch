package main

import (
	"fmt"
)

func main() {

	c := Comparator{}
	tree := NewTree(c)

	tree.MustPut("key1", "Val1")
	tree.MustPut("key2", "Val2")
	tree.MustPut("key3", "Val3")
	tree.MustPut("key4", "Val4")
	tree.MustPut("key5", "Val5")
	tree.MustPut("key6", "Val6")
	tree.MustPut("key7", "Val7")
	tree.MustPut("key8", "Val8")

	fmt.Println(tree.String())

	fmt.Println(tree.Ceil("key7"))
}
