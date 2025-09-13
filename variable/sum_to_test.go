package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleSumTo() {
	// p301
	x := variable.NewOf(
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	)
	y := variable.SumTo(0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)
	x.Cleargrad()

	y = variable.SumTo(1)(x)
	y.Backward()
	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3]([5 7 9])
	// variable[2 3]([1 1 1 1 1 1])
	// variable[2]([6 15])
	// variable[2 3]([1 1 1 1 1 1])
}
