package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleSumTo() {
	// p292
	x := variable.New(1, 2, 3, 4)
	y := variable.SumTo(1, 1)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable([10])
	// variable([1 1 1 1])
}
