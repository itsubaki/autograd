package variable_test

import (
	"fmt"
	"math"

	"github.com/itsubaki/autograd/variable"
)

func ExampleCosT() {
	x := variable.New(math.Pi / 4)
	f := variable.CosT{}

	fmt.Println(x)
	fmt.Println(f.Forward(x))
	fmt.Println(f.Backward(variable.OneLike(x)))

	// Output:
	// variable([0.7853981633974483])
	// [variable([0.7071067811865476])]
	// [variable([-0.7071067811865475])]
}

func ExampleCos() {
	// p198
	x := variable.New(math.Pi / 4)
	y := variable.Cos(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)
	fmt.Println([]float64{1.0 / math.Sqrt2})

	// Output:
	// variable([0.7071067811865476])
	// variable([-0.7071067811865475])
	// [0.7071067811865476]
}

func ExampleCos_double() {
	x := variable.New(1.0)
	y := variable.Cos(x)
	y.Backward(variable.Opts{CreateGraph: true})

	fmt.Println(y)
	fmt.Println(x.Grad)

	for i := 0; i < 10; i++ {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward(variable.Opts{CreateGraph: true})
		fmt.Println(x.Grad)
	}

	// Output:
	// variable([0.5403023058681398])
	// variable([-0.8414709848078965])
	// variable([-0.5403023058681398])
	// variable([0.8414709848078965])
	// variable([0.5403023058681398])
	// variable([-0.8414709848078965])
	// variable([-0.5403023058681398])
	// variable([0.8414709848078965])
	// variable([0.5403023058681398])
	// variable([-0.8414709848078965])
	// variable([-0.5403023058681398])
	// variable([0.8414709848078965])
}
