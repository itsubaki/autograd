package rand

import (
	randv2 "math/rand/v2"
)

// NewSource returns a source of pseudo-random number generator
func NewSource(seed [32]byte) randv2.Source {
	return randv2.NewChaCha8(seed)
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
