package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleClip() {
	x := variable.New(-2, -1, 0, 1, 2, 4, 6, 8)
	y := variable.Clip(0, 5)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[8]([0 0 0 1 2 4 5 5])
	// variable[8]([0 0 1 1 1 1 0 0])
}
