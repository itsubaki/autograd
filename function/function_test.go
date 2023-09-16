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
	// variable([1 2 3 4 5])
	// variable([1 4 9 16 25])
}

func ExampleData() {
	v := variable.New(1, 2, 3, 4, 5)
	w := variable.New(6, 7, 8, 9, 10)
	x := variable.New(11, 12, 13, 14, 15)

	data := F.Data(v, w, x)
	fmt.Println(data)

	// Output:
	// [[1 2 3 4 5] [6 7 8 9 10] [11 12 13 14 15]]
}
