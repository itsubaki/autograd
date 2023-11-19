package function_test

import (
	"fmt"
	"math/rand"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleDropoutSimple() {
	// p436
	s := rand.NewSource(1)

	x := variable.New(1, 1, 1, 1, 1)
	y := F.DropoutSimple(0.5, s)(x)
	fmt.Println(y)

	func() {
		defer variable.TestMode().End()

		y := F.DropoutSimple(0.5)(x)
		fmt.Println(y)
	}()

	// Output:
	// variable([2 2 2 0 0])
	// variable([1 1 1 1 1])
}

func ExampleDropoutSimple_backward() {
	s := rand.NewSource(1)
	x := variable.New(0.1, 0.2, 0.3, 0.4, 0.5)
	y := F.DropoutSimple(0.5, s)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable([0.2 0.4 0.6 0 0])
	// variable([2 2 2 0 0])
}
