package tensor

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"reflect"
)

const (
	NpyMagicString string = `\x93NUMPY`
	NpySuffix      string = ".npy"
)

func readHeader(filepath string) (string, error) {

	f, err := os.Open(filepath)
	if err != nil {
		return "", err
	}

	magicStr := make([]byte, len(NpyMagicString))
	r := bufio.NewReader(f)

	_, err = io.ReadFull(r, magicStr)
	if err != nil {
		return "", err
	}

	if string(magicStr) != NpyMagicString {
		err = fmt.Errorf("magic string mismatched.\n")
		return "", err
	}

	version := make([]byte, 2)

	_, err = io.ReadFull(r, version)
	if err != nil {
		return "", err
	}

	var headerLenLength int
	switch version[0] {
	case 1:
		headerLenLength = 2
	case 2:
		headerLenLength = 4
	default:
		err = fmt.Errorf("Unsupported version: %v\n", version[0])
	}

	headerLen := make([]byte, headerLenLength)

	_, err = io.ReadFull(r, headerLen)
	if err != nil {
		return "", err
	}

	var hLen int = 0
	for i := len(headerLen); i > 0; i-- {
		hLen = hLen*256 + int(headerLen[i])
	}

	header := make([]byte, hLen)
	_, err = io.ReadFull(r, header)
	if err != nil {
		return "", err
	}

	return string(header), nil
}

type Header struct {
	descr        reflect.Kind
	fortranOrder bool
	shape        []int64
}

func (h *Header) toString() (string, error) {
	var fortranOrder string = "False"

	if h.fortranOrder {
		fortranOrder = "True"
	}
}
