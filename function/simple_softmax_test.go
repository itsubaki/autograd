package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSoftmaxSimple() {
	x := variable.New(
		1, 2, 3,
		4, 4, 8,
	).Reshape(2, 3)

	y := F.SoftmaxSimple(x, 1)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2 3]([0.09003057 0.24472848 0.66524094 0.017668422 0.017668422 0.96466315])
	// variable[2 3]([0 0 0 0 0 0])
}

func ExampleSoftmaxSimple_axis1n() {
	x := variable.New(
		1, 2, 3,
		4, 4, 8,
	).Reshape(2, 3)

	y := F.SoftmaxSimple(x, -1)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[2 3]([0.09003057 0.24472848 0.66524094 0.017668422 0.017668422 0.96466315])
	// variable[2 3]([0 0 0 0 0 0])
}

func Example_softmax1d() {
	softmax1d := func(x *variable.Variable) *variable.Variable {
		y := F.Exp(x)
		sumy := F.Sum()(y)
		return F.Div(y, sumy)
	}

	x := variable.New(1, 2, 3)
	y := softmax1d(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3]([0.09003057 0.24472848 0.66524094])
	// variable[3]([0 0 0])
}

func ExampleSoftmaxSimple_double() {
	x := variable.New(
		1, 2, 3,
		4, 4, 8,
	).Reshape(2, 3)

	y := F.SoftmaxSimple(x, 1)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)
	fmt.Println(x.Grad)

	gx := x.Grad
	x.Cleargrad()
	gx.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[2 3]([0.09003057 0.24472848 0.66524094 0.017668422 0.017668422 0.96466315])
	// variable[2 3]([0 0 0 0 0 0])
	// variable[2 3]([0 0 0 -3.1780305e-09 -3.1780305e-09 -1.735146e-07])
}
