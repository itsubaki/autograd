package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExamplePowT() {
	x := variable.New(3.0)
	f := variable.PowT{C: 4.0}

	fmt.Println(x)
	fmt.Println(f.Forward(x))
	fmt.Println(f.Backward(variable.OneLike(x)))

	// Output:
	// variable([3])
	// [variable([81])]
	// [variable([108])]
}

func ExamplePow() {
	x := variable.New(2.0)
	y := variable.Pow(3.0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable([8])
	// variable([12])
}

func ExamplePow_double() {
	x := variable.New(2.0)
	y := variable.Pow(3.0)(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	for i := 0; i < 3; i++ {
		gx := x.Grad
		x.Cleargrad()
		y.Cleargrad()
		gx.Backward()
		fmt.Println(x.Grad)
	}

	// Output:
	// variable([8])
	// variable([12])
	// variable([12])
	// variable([6])
	// variable([0])
}
