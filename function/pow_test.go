package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExamplePow() {
	x := variable.New(2.0)
	y := F.Pow(3.0)(x)
	y[0].Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// [variable[8]]
	// variable[12]
}

func ExamplePowT() {
	x := variable.New(1, 2, 3, 4, 5)
	f := F.PowT{C: 3.0}

	fmt.Println(x)
	fmt.Println(f.Forward(x))
	fmt.Println(f.Backward(variable.OneLike(x)))

	// Output:
	// variable[1 2 3 4 5]
	// [variable[1 8 27 64 125]]
	// [variable[3 12 27 48 75]]
}

func ExamplePow_higher() {
	x := variable.New(2.0)
	y := F.Pow(3.0)(x)
	y[0].Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	for i := 0; i < 3; i++ {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward()
		fmt.Println(x.Grad)
	}

	// Output:
	// [variable[8]]
	// variable[12]
	// variable[12]
	// variable[6]
	// variable[0]
}
