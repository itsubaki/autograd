package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func Example_retaingrad() {
	// p121
	x0 := variable.New(1.0)
	x1 := variable.New(1.0)
	t := variable.Add(x0, x1)
	y := variable.Add(x0, t)
	y.Backward(variable.Opts{RetainGrad: true})

	fmt.Println(y.Grad, t.Grad)
	fmt.Println(x0.Grad, x1.Grad)

	// Output:
	// variable([1]) variable([1])
	// variable([2]) variable([1])
}

func Example_retaingrad_false() {
	// p123
	x0 := variable.New(1.0)
	x1 := variable.New(1.0)
	t := variable.Add(x0, x1)
	y := variable.Add(x0, t)
	y.Backward()

	fmt.Println(y.Grad, t.Grad)
	fmt.Println(x0.Grad, x1.Grad)

	// Output:
	// <nil> <nil>
	// variable([2]) variable([1])
}
