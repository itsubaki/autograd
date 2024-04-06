package rand

import (
	crand "crypto/rand"
	"math/rand/v2"
)

// NewSource returns a source of pseudo-random number generator
func NewSource() rand.Source {
	var p [32]byte
	if _, err := crand.Read(p[:]); err != nil {
		panic(err)
	}

	return rand.NewChaCha8(p)
}

// Const returns a source of constant pseudo-random number generator
func Const(seed ...uint64) rand.Source {
	var s0, s1 uint64
	if len(seed) > 0 {
		s0 = seed[0]
	}

	if len(seed) > 1 {
		s1 = seed[1]
	}

	return rand.NewPCG(s0, s1)
}
