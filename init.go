package gotch

import (
	"fmt"
	"log"
	"os"
	"strconv"
)

var (
	CachedDir     string = "NOT_SETTING"
	gotchEnvKey   string = "GOTCH_CACHE"
	gotchDebugKey string = "GOTCH_DEBUG"
	Debug         bool   = false
)

func init() {
	homeDir := os.Getenv("HOME")
	CachedDir = fmt.Sprintf("%s/.cache/gotch", homeDir) // default dir: "{$HOME}/.cache/gotch"

	initEnv()
}

func initEnv() {
	if v, err := strconv.ParseBool(os.Getenv(gotchDebugKey)); err == nil {
		Debug = v
	}

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
