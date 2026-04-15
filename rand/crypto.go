package rand

import (
	"crypto/rand"
	"fmt"
)

var RandRead = rand.Read

// Must returns a or panics if err is non-nil.
func Must[T any](a T, err error) T {
	if err != nil {
		panic(err)
	}

	return a
}

// MustRead reads 32 random bytes and panics on error.
func MustRead() [32]byte {
	return Must(Read())
}

// Read reads 32 random bytes from the crypto/rand reader.
func Read() ([32]byte, error) {
	var p [32]byte
	if _, err := RandRead(p[:]); err != nil {
		return [32]byte{}, fmt.Errorf("read: %w", err)
	}

	return p, nil
}
