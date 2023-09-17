package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleFunction() {
	f := &F.Function{
		Forwarder: &F.SquareT{},
	}

	v := variable.New(1, 2, 3, 4, 5)
	f.Apply(v)

	fmt.Println(f.Input())
	fmt.Println(f.Output())

	// Output:
	// [variable([1 2 3 4 5])]
	// [variable([1 4 9 16 25])]
}
