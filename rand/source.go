package rand

import (
	crand "crypto/rand"
	"fmt"
	randv2 "math/rand/v2"
)

func Must[T any](a T, err error) T {
	if err != nil {
		panic(err)
	}

	return a
}

func MustNewSource() randv2.Source {
	return Must(NewSource())
}

// NewSource returns a source of pseudo-random number generator
func NewSource() (randv2.Source, error) {
	var p [32]byte
	if _, err := crand.Read(p[:]); err != nil {
		return nil, fmt.Errorf("read: %v", err)
	}

	return randv2.NewChaCha8(p), nil
}

// Const returns a source of constant pseudo-random number generator
func Const(seed ...uint64) randv2.Source {
	var s0, s1 uint64
	if len(seed) > 0 {
		s0 = seed[0]
	}

	if len(seed) > 1 {
		s1 = seed[1]
	}

	return randv2.NewPCG(s0, s1)
}
