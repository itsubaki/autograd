package rand_test

import (
	"fmt"
	randv2 "math/rand/v2"
	"testing"

	"github.com/itsubaki/autograd/rand"
)

func TestMustRead(t *testing.T) {
	v := randv2.New(rand.NewSource(rand.MustRead())).Float64()
	if v >= 0 && v < 1 {
		return
	}

	t.Fail()
}

func TestMustPanic(t *testing.T) {
	defer func() {
		if rec := recover(); rec != nil {
			err, ok := rec.(error)
			if !ok {
				t.Fail()
			}

			if err.Error() != "something went wrong" {
				t.Fail()
			}
		}
	}()

	rand.Must(-1, fmt.Errorf("something went wrong"))
	t.Fail()
}
