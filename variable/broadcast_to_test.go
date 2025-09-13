package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleBroadcastTo() {
	x := variable.New(2)
	y := variable.BroadcastTo(1, 3)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[1 3]([2 2 2])
	// variable(3)
}
