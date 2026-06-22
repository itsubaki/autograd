package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleUnsqueeze() {
	x := variable.New(
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
	).Reshape(3, 3)

	y := variable.Unsqueeze(1)(x)
	y.Backward()

	fmt.Println(x)
	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3 3]([1 2 3 4 5 6 7 8 9])
	// variable[3 1 3]([1 2 3 4 5 6 7 8 9])
	// variable[3 3]([1 1 1 1 1 1 1 1 1])
}
