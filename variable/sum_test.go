package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleSum() {
	// p292
	x := variable.New(1, 2, 3, 4, 5, 6)
	y := variable.Sum(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[21]
	// variable[1 1 1 1 1 1]
}
