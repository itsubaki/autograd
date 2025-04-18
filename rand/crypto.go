package rand

import (
	"crypto/rand"
	"fmt"
)

var RandRead = rand.Read

func Must[T any](a T, err error) T {
	if err != nil {
		panic(err)
	}

	return a
}

func MustRead() [32]byte {
	return Must(Read())
}

func Read() ([32]byte, error) {
	var p [32]byte
	if _, err := RandRead(p[:]); err != nil {
		return [32]byte{}, fmt.Errorf("read: %w", err)
	}

	return p, nil
}
