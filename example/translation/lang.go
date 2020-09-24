package main

import (
	"strings"
)

const (
	SosToken = "SOS"
	EosToken = "EOS"
)

type IndexCount struct {
	Index int
	Count int
}

type Lang struct {
	Name                string
	WordToIndexAndCount map[string]IndexCount
	IndexToWord         map[int]string
}

func NewLang(name string) (retVal Lang) {

	lang := Lang{
		Name:                name,
		WordToIndexAndCount: make(map[string]IndexCount, 0),
		IndexToWord:         make(map[int]string, 0),
	}

	lang.AddWord(SosToken)
	lang.AddWord(EosToken)

	return lang
}

func (l *Lang) AddWord(word string) {
	if len(word) > 0 {
		idxCount, ok := l.WordToIndexAndCount[word]
		if !ok {
			length := len(l.WordToIndexAndCount)
			l.WordToIndexAndCount[word] = IndexCount{length, 1}
			l.IndexToWord[length] = word
		} else {
			idxCount.Count += 1
			l.WordToIndexAndCount[word] = idxCount
		}
	}
}

func (l *Lang) AddSentence(sentence string) {
	words := strings.Split(sentence, " ")
	for _, word := range words {
		l.AddWord(word)
	}
}

func (l *Lang) Len() (retVal int) {
	return len(l.IndexToWord)
}

func (l *Lang) SosToken() (retVal int) {
	return l.WordToIndexAndCount[SosToken].Index
}

func (l *Lang) EosToken() (retVal int) {
	return l.WordToIndexAndCount[EosToken].Index
}

func (l *Lang) GetName() (retVal string) {
	return l.Name
}

func (l *Lang) GetIndex(word string) (retVal int) {
	idxCount, ok := l.WordToIndexAndCount[word]
	if ok {
		return idxCount.Index
	} else {
		return -1 // word does not exist in Lang
	}
}

func (l *Lang) SeqToString(seq []int) (retVal string) {
	var words []string = make([]string, 0)

	for _, idx := range seq {
		w := l.IndexToWord[idx]
		words = append(words, w)
	}

	return strings.Join(words, " ")
}
