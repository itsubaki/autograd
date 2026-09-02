package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleMinimum() {
	x0 := variable.New(
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
	)

	x1 := variable.New(
		0.3, 0.2, 0.1,
		0.6, 0.5, 0.4,
	)

	y := F.Minimum(x0, x1)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x0.Grad)
	fmt.Println(x1.Grad)

	// Output:
	// variable[6]([0.1 0.2 0.1 0.4 0.5 0.4])
	// variable[6]([1 1 0 1 1 0])
	// variable[6]([0 0 1 0 0 1])
}
