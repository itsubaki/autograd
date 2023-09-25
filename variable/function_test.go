package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleFunction() {
	f := &variable.Function{
		Forwarder: &variable.SinT{},
	}

	y := f.Apply(variable.New(1.0))
	fmt.Println(f)
	fmt.Println(y)

	// Output:
	// *variable.SinT[variable[1]]
	// [variable[0.8414709848078965]]
}
