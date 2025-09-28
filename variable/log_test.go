package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleLogT() {
	x := variable.New(1, 2, 3, 4, 5)
	f := variable.LogT{}

	fmt.Println(x)
	fmt.Println(f.Forward(x))
	fmt.Println(f.Backward(variable.OneLike(x)))

	// Output:
	// variable[5]([1 2 3 4 5])
	// [variable[5]([0 0.6931471805599453 1.0986122886681096 1.3862943611198906 1.6094379124341003])]
	// [variable[5]([1 0.5 0.3333333333333333 0.25 0.2])]
}

func ExampleLog() {
	v := variable.New(1, 2, 3, 4, 5)
	y := variable.Log(v)
	y.Backward()

	fmt.Println(v.Grad)

	// Output:
	// variable[5]([1 0.5 0.3333333333333333 0.25 0.2])
}

func ExampleLog_double() {
	x := variable.New(2)

	y := variable.Log(x)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)
	fmt.Println(x.Grad)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(x.Grad)

	gx = x.Grad
	x.Cleargrad()
	gx.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(x.Grad)

	gx = x.Grad
	x.Cleargrad()
	gx.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(x.Grad)

	// Output:
	// variable(0.6931471805599453)
	// variable(0.5)
	// variable(-0.25)
	// variable(0.25)
	// variable(-0.375)
}
