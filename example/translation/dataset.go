package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"unicode"
)

var Prefixes []string = []string{
	"i am ",
	"i m ",
	"you are ",
	"you re",
	"he is ",
	"he s ",
	"she is ",
	"she s ",
	"we are ",
	"we re ",
	"they are ",
	"they re ",
}

type Pair struct {
	val1 string
	val2 string
}

type Dataset struct {
	inputLang  Lang
	outputLang Lang
	pairs      []Pair
}

func normalize(s string) (retVal string) {

	lower := strings.ToLower(s)
	/*
	 *   // strip all spaces
	 *   noSpaceStr := strings.Map(func(r rune) rune {
	 *     if unicode.IsSpace(r) {
	 *       return -1
	 *     }
	 *     return 1
	 *   }, lower)
	 *  */
	// add single space before "!", ".", "?"
	var res []rune
	for _, r := range []rune(lower) {
		char := fmt.Sprintf("%c", r)
		switch {
		case char == "!":
			res = append(res, []rune(" !")...)
		case char == ".":
			res = append(res, []rune(" .")...)
		case char == "?":
			res = append(res, []rune(" ?")...)
		case unicode.IsLetter(r), unicode.IsNumber(r):
			res = append(res, r)
		default:
			res = append(res, []rune(" ")...)
		}
	}

	return string(res)
}

func toIndexes(s string, lang Lang) (retVal []int) {
	res := strings.Split(s, " ")

	for _, l := range res {
		idx := lang.GetIndex(l)
		if idx >= 0 {
			retVal = append(retVal, idx)
		}
	}

	retVal = append(retVal, lang.EosToken())

	return retVal
}

func filterPrefix(s string) (retVal bool) {

	for _, prefix := range Prefixes {
		if strings.HasPrefix(s, prefix) {
			return true
		}
	}

	return false
}

func readPairs(ilang, olang string, maxLength int) (retVal []Pair) {
	file, err := os.Open(fmt.Sprintf("../../data/translation/%v-%v.txt", ilang, olang))
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()

		if strings.Contains(line, "\t") {
			// NOTE: assuming there's only 1 '\t'
			pair := strings.Split(line, "\t")
			lhs := normalize(pair[0])
			rhs := normalize(pair[1])

			if (len(strings.Split(lhs, " ")) < maxLength) && (len(strings.Split(rhs, " ")) < maxLength) && (filterPrefix(lhs) || filterPrefix(rhs)) {
				retVal = append(retVal, Pair{lhs, rhs})
			}

		} else {
			log.Fatalf("A line does not contain a single tab: %v\n", line)
		}

	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	return retVal
}

func newDataset(ilang, olang string, maxLength int) (retVal Dataset) {
	pairs := readPairs(ilang, olang, maxLength)
	inputLang := NewLang(ilang)
	outputLang := NewLang(olang)

	for _, p := range pairs {
		inputLang.AddSentence(p.val1)
		outputLang.AddSentence(p.val2)
	}

	return Dataset{
		inputLang:  inputLang,
		outputLang: outputLang,
		pairs:      pairs,
	}
}

func (ds Dataset) InputLang() (retVal Lang) {
	return ds.inputLang
}

func (ds Dataset) OutputLang() (retVal Lang) {
	return ds.outputLang
}

func (ds Dataset) Reverse() (retVal Dataset) {
	var rpairs []Pair
	for _, p := range ds.pairs {
		rpairs = append(rpairs, Pair{p.val2, p.val1})
	}
	return Dataset{
		inputLang:  ds.outputLang,
		outputLang: ds.inputLang,
		pairs:      rpairs,
	}
}

type Pairs struct {
	Val1 []int
	Val2 []int
}

func (ds Dataset) Pairs() (retVal []Pairs) {
	for _, p := range ds.pairs {
		val1 := toIndexes(p.val1, ds.inputLang)
		val2 := toIndexes(p.val2, ds.outputLang)

		retVal = append(retVal, Pairs{val1, val2})
	}

	return retVal
}
