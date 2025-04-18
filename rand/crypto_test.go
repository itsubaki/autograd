package rand_test

import (
	crand "crypto/rand"
	"errors"
	"fmt"
	randv2 "math/rand/v2"
	"testing"

	"github.com/itsubaki/autograd/rand"
)

var ErrSomtingWentWrong = errors.New("something went wrong")

func ExampleRead() {
	defer func() {
		rand.RandRead = crand.Read
	}()

	rand.RandRead = func(b []byte) (int, error) {
		return 0, ErrSomtingWentWrong
	}

	if _, err := rand.Read(); err != nil {
		fmt.Println(err)
	}

	// Output:
	// read: something went wrong
}

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

			if !errors.Is(err, ErrSomtingWentWrong) {
				t.Fail()
			}
		}
	}()

	rand.Must(-1, ErrSomtingWentWrong)
	t.Fail()
}
