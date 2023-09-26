package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleBroadcast() {
	x := variable.New(2)
	y := variable.Broadcast(3)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2 2 2]
	// variable[3]
}
