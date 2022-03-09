package gotch

import (
	"fmt"
	"log"
	"os"
)

var (
	CachedDir   string = "NOT_SETTING"
	gotchEnvKey string = "GOTCH_CACHE"
)

func init() {
	homeDir := os.Getenv("HOME")
	CachedDir = fmt.Sprintf("%s/.cache/gotch", homeDir) // default dir: "{$HOME}/.cache/gotch"

	initEnv()
	// log.Printf("INFO: CacheDir=%q\n", CacheDir)
}

func initEnv() {
	val := os.Getenv(gotchEnvKey)
	if val != "" {
		CachedDir = val
	}

	if _, err := os.Stat(CachedDir); os.IsNotExist(err) {
		if err := os.MkdirAll(CachedDir, 0755); err != nil {
			log.Fatal(err)
		}
	}
}
