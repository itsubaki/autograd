package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleReLU() {
	x := variable.NewOf(
		[]float64{1, -2, 3},
		[]float64{-4, 5, -6},
	)

	y := F.ReLU(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable([[1 0 3] [0 5 0]])
	// variable([[1 0 1] [0 1 0]])
}
