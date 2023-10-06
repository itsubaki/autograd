package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleTranspose() {
	// p286
	x := variable.NewOf([]float64{1, 2, 3}, []float64{4, 5, 6})
	y := variable.Transpose(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable([[1 4] [2 5] [3 6]])
	// variable([[1 1 1] [1 1 1]])
}
