package gotch

import (
	"fmt"
	"log"
	"os"
)

var (
	CacheDir    string = "NOT_SETTING"
	gotchEnvKey string = "GOTCH_CACHE"
)

func init() {
	homeDir := os.Getenv("HOME")
	CacheDir = fmt.Sprintf("%s/.cache/gotch", homeDir) // default dir: "{$HOME}/.cache/gotch"

	initEnv()
	// log.Printf("INFO: CacheDir=%q\n", CacheDir)
}

func initEnv() {
	val := os.Getenv(gotchEnvKey)
	if val != "" {
		CacheDir = val
	}

	if _, err := os.Stat(CacheDir); os.IsNotExist(err) {
		if err := os.MkdirAll(CacheDir, 0755); err != nil {
			log.Fatal(err)
		}
	}
}
