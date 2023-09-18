package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleAdd() {
	v := variable.New(2, 3)
	w := variable.New(3, 4)

	y := F.Add(v, w)
	y.Backward()

	fmt.Println(v.Grad)
	fmt.Println(w.Grad)

	// Output:
	// variable[1 1]
	// variable[1 1]
}

func ExampleAddT() {
	v := variable.New(2, 3)
	w := variable.New(3, 4)
	fmt.Println(v)
	fmt.Println(w)

	f := F.AddT{}
	fmt.Println(f.Forward(v.Data, w.Data))
	fmt.Println(f.Backward(variable.OneLike(v), variable.OneLike(w)))

	// Output:
	// variable[2 3]
	// variable[3 4]
	// [[5 7]]
	// [variable[1 1] variable[1 1]]
}
