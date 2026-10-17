package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleExpT() {
	x := variable.New(1, 2, 3, 4, 5)
	f := variable.ExpT{}

	fmt.Println(x)
	fmt.Println(f.Forward(x))
	fmt.Println(f.Backward(variable.OnesLike(x)))

	// Output:
	// variable[5]([1 2 3 4 5])
	// [variable[5]([2.7182817 7.389056 20.085537 54.59815 148.41316])]
	// [variable[5]([2.7182817 7.389056 20.085537 54.59815 148.41316])]
}

func ExampleExp() {
	v := variable.New(1, 2, 3, 4, 5)
	y := variable.Exp(v)
	y.Backward()

	fmt.Println(v.Grad)

	// Output:
	// variable[5]([2.7182817 7.389056 20.085537 54.59815 148.41316])
}

func ExampleExp_double() {
	x := variable.New(2.0)

	y := variable.Exp(x)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)
	fmt.Println(x.Grad)

	for i := 0; i < 3; i++ {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward(variable.Opts{CreateGraph: true})
		fmt.Println(x.Grad)
	}

	// Output:
	// variable(7.389056)
	// variable(7.389056)
	// variable(7.389056)
	// variable(7.389056)
	// variable(7.389056)
}
