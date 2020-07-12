package main

import (
	"encoding/json"
	"fmt"
	"log"
	"strconv"
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
	return &Tree{
		Root:       nil,
		Comparator: c,
		size:       0,
	}
}

// Put inserts a node to tree
func (t *Tree) Put(k interface{}, v interface{}) (err error) {
	var insertedNode *Node

	if t.Root == nil {
		t.Comparator = t.Comparator.Init()
		t.Root = &Node{Key: k, Value: v, color: red}
		insertedNode = t.Root
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

// Left returns the left-most (min) node or nil if tree is empty.
func (t *Tree) Left() *Node {
	var parent *Node
	current := t.Root
	for current != nil {
		parent = current
		current = current.Left
	}
	return parent
}

// Right returns the right-most (max) node or nil if tree is empty.
func (t *Tree) Right() *Node {
	var parent *Node
	current := t.Root
	for current != nil {
		parent = current
		current = current.Right
	}
	return parent
}

// Floor finds floor node of the input key.
//
// NOTE: `floor node` is defined as the largest node that is smaller or equal
// to given node. There may be no floor node if the tree is empty or all nodes
// in the tree are larger than the given node.
func (t *Tree) Floor(k interface{}) (retVal *Node, found bool) {
	found = false
	root := t.Root

	for root != nil {
		switch t.Comparator.Compare(k, root.Key) {
		case 0:
			return root, true
		case -1:
			root = root.Left
		case 1:
			retVal, found = root, true
			root = root.Right
		}
	}

	if !found {
		return nil, false
	}

	return retVal, found
}

// Ceil returns the ceiling node of the input node.
//
// Ceiling node is defined as the smallest node that is larger or
// equal to the given node. There may not have a ceiling node if
// tree is empty or all other nodes in the tree are smaller than
// the given node.
func (t *Tree) Ceil(k interface{}) (retVal *Node, found bool) {
	found = false
	root := t.Root

	for root != nil {
		switch t.Comparator.Compare(k, root.Key) {
		case 0:
			return root, true
		case -1:
			retVal, found = root, true
			root = root.Left
		case 1:
			root = root.Right
		}
	}

	if !found {
		return nil, false
	}

	return retVal, found
}

// Clear deletes all nodes
func (t *Tree) Clear() {
	t.Root = nil
	t.size = 0
}

// String returns a string representation of the tree.
func (t *Tree) String() (retVal string) {
	str := "BinaryTree\n"

	if !t.IsEmpty() {
		output(t.Root, "", true, &str)
	}

	return str
}

// Node methods:
//==============

func (n *Node) String() (retVal string) {
	return fmt.Sprintf("%v", n.Key)
}

func output(n *Node, prefix string, isTail bool, str *string) {
	if n.Right != nil {
		newPrefix := prefix
		if isTail {
			newPrefix += "│   "
		} else {
			newPrefix += "    "
		}
		output(n.Right, newPrefix, false, str)
	}
	*str += prefix
	if isTail {
		*str += "└── "
	} else {
		*str += "┌── "
	}
	*str += n.String() + "\n"
	if n.Left != nil {
		newPrefix := prefix
		if isTail {
			newPrefix += "    "
		} else {
			newPrefix += "│   "
		}
		output(n.Left, newPrefix, true, str)
	}
}

func (t *Tree) lookup(k interface{}) (retVal *Node) {
	root := t.Root

	for root != nil {
		switch t.Comparator.Compare(k, root.Key) {
		case 0:
			return root
		case -1:
			root = root.Left
		case 1:
			root = root.Right
		}
	}

	return nil
}

func (n *Node) grandparent() (retVal *Node) {
	if n != nil && n.Parent != nil {
		return n.Parent.Parent
	}

	return nil
}

func (n *Node) uncle() (retVal *Node) {
	if n == nil || n.Parent == nil || n.Parent.Parent == nil {
		return nil
	}

	return n.Parent.sibling()
}

func (n *Node) sibling() (retVal *Node) {
	if n == nil || n.Parent == nil {
		return nil
	}

	if n == n.Parent.Left {
		return n.Parent.Right
	}

	return n.Parent.Left
}

func (t *Tree) rotateLeft(n *Node) {
	right := n.Right
	t.replaceNode(n, right)

	n.Right = right.Left

	if right.Left != nil {
		right.Left.Parent = n
	}

	right.Left = n
	n.Parent = right
}

func (t *Tree) rotateRight(n *Node) {
	left := n.Left
	t.replaceNode(n, left)
	n.Left = left.Right

	if left.Right != nil {
		left.Right.Parent = n
	}

	left.Right = n
	n.Parent = left
}

func (t *Tree) replaceNode(old, new *Node) {
	if old.Parent == nil {
		t.Root = new
	} else {
		if old == old.Parent.Left {
			old.Parent.Left = new
		} else {
			old.Parent.Right = new
		}
	}

	if new != nil {
		new.Parent = old.Parent
	}
}

func (t *Tree) insertCase1(node *Node) {
	if node.Parent == nil {
		node.color = black
	} else {
		t.insertCase2(node)
	}
}

func (t *Tree) insertCase2(n *Node) {
	if nodeColor(n.Parent) == black {
		return
	}
	t.insertCase3(n)
}

func (t *Tree) insertCase3(n *Node) {
	uncle := n.uncle()
	if nodeColor(uncle) == red {
		n.Parent.color = black
		uncle.color = black
		n.grandparent().color = red
		t.insertCase1(n.grandparent())
	} else {
		t.insertCase4(n)
	}
}

func (t *Tree) insertCase4(n *Node) {
	grandparent := n.grandparent()
	if n == n.Parent.Right && n.Parent == grandparent.Left {
		t.rotateLeft(n.Parent)
		n = n.Left
	} else if n == n.Parent.Left && n.Parent == grandparent.Right {
		t.rotateRight(n.Parent)
		n = n.Right
	}
	t.insertCase5(n)
}

func (t *Tree) insertCase5(n *Node) {
	n.Parent.color = black
	grandparent := n.grandparent()
	grandparent.color = red
	if n == n.Parent.Left && n.Parent == grandparent.Left {
		t.rotateRight(grandparent)
	} else if n == n.Parent.Right && n.Parent == grandparent.Right {
		t.rotateLeft(grandparent)
	}
}

func (n *Node) maximumNode() (retVal *Node) {
	if n == nil {
		return nil
	}
	for n.Right != nil {
		n = n.Right
	}
	return n
}

func (t *Tree) deleteCase1(n *Node) {
	if n.Parent == nil {
		return
	}
	t.deleteCase2(n)
}

func (t *Tree) deleteCase2(n *Node) {
	sibling := n.sibling()
	if nodeColor(sibling) == red {
		n.Parent.color = red
		sibling.color = black
		if n == n.Parent.Left {
			t.rotateLeft(n.Parent)
		} else {
			t.rotateRight(n.Parent)
		}
	}
	t.deleteCase3(n)
}

func (t *Tree) deleteCase3(n *Node) {
	sibling := n.sibling()
	if nodeColor(n.Parent) == black &&
		nodeColor(sibling) == black &&
		nodeColor(sibling.Left) == black &&
		nodeColor(sibling.Right) == black {
		sibling.color = red
		t.deleteCase1(n.Parent)
	} else {
		t.deleteCase4(n)
	}
}

func (t *Tree) deleteCase4(n *Node) {
	sibling := n.sibling()
	if nodeColor(n.Parent) == red &&
		nodeColor(sibling) == black &&
		nodeColor(sibling.Left) == black &&
		nodeColor(sibling.Right) == black {
		sibling.color = red
		n.Parent.color = black
	} else {
		t.deleteCase5(n)
	}
}

func (t *Tree) deleteCase5(n *Node) {
	sibling := n.sibling()
	if n == n.Parent.Left &&
		nodeColor(sibling) == black &&
		nodeColor(sibling.Left) == red &&
		nodeColor(sibling.Right) == black {
		sibling.color = red
		sibling.Left.color = black
		t.rotateRight(sibling)
	} else if n == n.Parent.Right &&
		nodeColor(sibling) == black &&
		nodeColor(sibling.Right) == red &&
		nodeColor(sibling.Left) == black {
		sibling.color = red
		sibling.Right.color = black
		t.rotateLeft(sibling)
	}
	t.deleteCase6(n)
}

func (t *Tree) deleteCase6(n *Node) {
	sibling := n.sibling()
	sibling.color = nodeColor(n.Parent)
	n.Parent.color = black
	if n == n.Parent.Left && nodeColor(sibling.Right) == red {
		sibling.Right.color = black
		t.rotateLeft(n.Parent)
	} else if nodeColor(sibling.Left) == red {
		sibling.Left.color = black
		t.rotateRight(n.Parent)
	}
}

func nodeColor(n *Node) bool {
	if n == nil {
		return black
	}
	return n.color
}

// ToJSON outputs the JSON representation of the tree.
func (t *Tree) ToJSON() ([]byte, error) {
	elements := make(map[string]interface{})
	it := t.Iterator()
	for it.Next() {
		elements[toString(it.Key())] = it.Value()
	}
	return json.Marshal(&elements)
}

// FromJSON populates the tree from the input JSON representation.
func (t *Tree) FromJSON(data []byte) error {
	elements := make(map[string]interface{})
	err := json.Unmarshal(data, &elements)
	if err == nil {
		t.Clear()
		for key, value := range elements {
			t.Put(key, value)
		}
	}
	return err
}

// ToString converts a value to string.
func toString(value interface{}) string {
	switch value := value.(type) {
	case string:
		return value
	case int8:
		return strconv.FormatInt(int64(value), 10)
	case int16:
		return strconv.FormatInt(int64(value), 10)
	case int32:
		return strconv.FormatInt(int64(value), 10)
	case int64:
		return strconv.FormatInt(int64(value), 10)
	case uint8:
		return strconv.FormatUint(uint64(value), 10)
	case uint16:
		return strconv.FormatUint(uint64(value), 10)
	case uint32:
		return strconv.FormatUint(uint64(value), 10)
	case uint64:
		return strconv.FormatUint(uint64(value), 10)
	case float32:
		return strconv.FormatFloat(float64(value), 'g', -1, 64)
	case float64:
		return strconv.FormatFloat(float64(value), 'g', -1, 64)
	case bool:
		return strconv.FormatBool(value)
	default:
		return fmt.Sprintf("%+v", value)
	}
}
