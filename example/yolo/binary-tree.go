package main

import (
	"fmt"
	"log"
)

const (
	black, red bool = true, false
)

// Tree is a tree container that holds TNode elements
type Tree struct {
	Root       *Node
	size       int
	Comparator Comparator
}

// Node is a tree element
type Node struct {
	Key    interface{}
	Value  interface{}
	Left   *Node
	Right  *Node
	Parent *Node
	color  bool
}

// NewTree creates a tree with a specified comparator
func NewTree(c Comparator) *Tree {
	return &Tree{Comparator: c}
}

// Put inserts a node to tree
func (t *Tree) Put(k interface{}, v interface{}) (err error) {
	var insertedNode *Node

	if t.Root == nil {
		t.Comparator = t.Comparator.Init()
		t.Root = &Node{Key: k, Value: v, color: red}
		err = nil
	} else {
		node := t.Root

		var loop bool = true
		for loop {

			switch t.Comparator.Compare(k, node.Key) {
			case 0:
				node.Key = k
				node.Value = v
				return
			case -1:
				if node.Left == nil {
					node.Left = &Node{Key: k, Value: v, color: red}
					insertedNode = node.Left
					loop = false
				} else {
					node = node.Left
				}
			case 1:
				if node.Right == nil {
					node.Right = &Node{Key: k, Value: v, color: red}
					insertedNode = node.Right
					loop = false
				} else {
					node = node.Right
				}
			}
		} // end of for loop

		insertedNode.Parent = node
	}

	t.insertCase1(insertedNode)
	t.size++

	return nil
}

// MustPut inserts new node. It will panic if err occurs.
func (t *Tree) MustPut(k interface{}, v interface{}) {
	err := t.Put(k, v)
	if err != nil {
		log.Fatal(err)
	}
}

// Get returns a corresponding value for specified key. If not found, returns
// nil.
func (t *Tree) Get(k interface{}) (retVal interface{}, found bool) {
	node := t.lookup(k)
	if node == nil {
		return nil, false
	}

	return node.Value, true
}

func (t *Tree) Remove(k interface{}) (err error) {
	var child *Node

	node := t.lookup(k)
	if node == nil {
		err = fmt.Errorf("Node not found for specified key (%v)\n", k)
		return err
	}

	if node.Left != nil && node.Right != nil {
		pred := node.maximumNode()
		node.Key = pred.Key
		node.Value = pred.Value
		node = pred
	}

	if node.Left != nil || node.Right != nil {
		if node.Right == nil {
			child = node.Left
		} else {
			child = node.Right
		}

		if node.color == black {
			node.color = nodeColor(child)
			t.deleteCase1(node)
		}

		t.replaceNode(node, child)

		if node.Parent == nil && child != nil {
			child.color = black
		}
	}

	t.size--

	return nil
}

// IsEmpty returns whether tree is empty
func (t *Tree) IsEmpty() (retVal bool) {
	return t.size == 0
}

// Size returns number of nodes in tree
func (t *Tree) Size() (retVal int) {
	return t.size
}

// Keys returns all keys (in order)
func (t *Tree) Keys() (retVal []interface{}) {
	keys := make([]interface{}, t.size)
	iter := t.Iterator()
	for i := 0; iter.Next(); i++ {
		keys[i] = iter.Key()
	}
	return keys
}

// Values returns all values of tree
func (t *Tree) Values() (retVal []interface{}) {
	vals := make([]interface{}, t.size)
	iter := t.Iterator()
	for i := 0; iter.Next(); i++ {
		vals[i] = iter.Value()
	}

	return vals
}

func (t *Tree) insertCase1(node *Node) {
	if node.Parent == nil {
		node.color = black
	} else {
		t.insertCase2(node)
	}
}

func (tree *Tree) insertCase2(node *Node) {
	if nodeColor(node.Parent) == black {
		return
	}
	tree.insertCase3(node)
}

func (tree *Tree) insertCase3(node *Node) {
	uncle := node.uncle()
	if nodeColor(uncle) == red {
		node.Parent.color = black
		uncle.color = black
		node.grandparent().color = red
		tree.insertCase1(node.grandparent())
	} else {
		tree.insertCase4(node)
	}
}

func (tree *Tree) insertCase4(node *Node) {
	grandparent := node.grandparent()
	if node == node.Parent.Right && node.Parent == grandparent.Left {
		tree.rotateLeft(node.Parent)
		node = node.Left
	} else if node == node.Parent.Left && node.Parent == grandparent.Right {
		tree.rotateRight(node.Parent)
		node = node.Right
	}
	tree.insertCase5(node)
}

func (tree *Tree) insertCase5(node *Node) {
	node.Parent.color = black
	grandparent := node.grandparent()
	grandparent.color = red
	if node == node.Parent.Left && node.Parent == grandparent.Left {
		tree.rotateRight(grandparent)
	} else if node == node.Parent.Right && node.Parent == grandparent.Right {
		tree.rotateLeft(grandparent)
	}
}

func nodeColor(node *Node) bool {
	if node == nil {
		return black
	}
	return node.color
}
