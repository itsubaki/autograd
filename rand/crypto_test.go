package rand_test

import (
	crand "crypto/rand"
	"fmt"
	randv2 "math/rand/v2"
	"strings"
	"testing"

	"github.com/itsubaki/autograd/rand"
)

func ExampleRead() {
	reader := crand.Reader
	defer func() {
		crand.Reader = reader
	}()

	crand.Reader = strings.NewReader("io.Reader stream to be read\n")
	if _, err := rand.Read(); err != nil {
		fmt.Println(err)
	}

	// Output:
	// read: unexpected EOF
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

			if err.Error() != "something went wrong" {
				t.Fail()
			}
		}
	}()

	rand.Must(-1, fmt.Errorf("something went wrong"))
	t.Fail()
}
