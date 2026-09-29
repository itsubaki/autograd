package variable_test

import (
	"fmt"
	"math"

	"github.com/itsubaki/autograd/variable"
)

func ExampleSin() {
	// p198
	x := variable.New(math.Pi / 4)
	y := variable.Sin(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)
	fmt.Println(1.0 / math.Sqrt2)

	// Output:
	// variable(0.70710677)
	// variable(0.70710677)
	// 0.7071067811865476
}

func ExampleSinT() {
	x := variable.New(math.Pi / 4)
	f := variable.SinT{}

	fmt.Println(x)
	fmt.Println(f.Forward(x))
	fmt.Println(f.Backward(variable.OnesLike(x)))

	// Output:
	// variable(0.7853982)
	// [variable(0.70710677)]
	// [variable(0.70710677)]
}

func ExampleSin_double() {
	// p243
	x := variable.New(1.0)
	y := variable.Sin(x)
	y.Backward(variable.Opts{CreateGraph: true})

	fmt.Println(y)
	fmt.Println(x.Grad)

	for range 10 {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward(variable.Opts{CreateGraph: true})

		fmt.Println(x.Grad)
	}

	// Output:
	// variable(0.84147096)
	// variable(0.5403023)
	// variable(-0.84147096)
	// variable(-0.5403023)
	// variable(0.84147096)
	// variable(0.5403023)
	// variable(-0.84147096)
	// variable(-0.5403023)
	// variable(0.84147096)
	// variable(0.5403023)
	// variable(-0.84147096)
	// variable(-0.5403023)
}
