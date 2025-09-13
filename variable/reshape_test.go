package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleReshape() {
	// p282
	x := variable.NewOf([]float64{1, 2, 3}, []float64{4, 5, 6})
	y := variable.Reshape(1, 6)(x)
	y.Backward(variable.Opts{RetainGrad: true})

	fmt.Println(x)
	fmt.Println(y)
	fmt.Println(x.Grad)
	fmt.Println(y.Grad)

	// Output:
	// variable[2 3]([[1 2 3] [4 5 6]])
	// variable[1 6]([1 2 3 4 5 6])
	// variable[2 3]([[1 1 1] [1 1 1]])
	// variable[1 6]([1 1 1 1 1 1])
}
