package function_test

import (
	"fmt"
	"math"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSin() {
	// p198
	x := variable.New(math.Pi / 4)
	y := F.Sin(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)
	fmt.Println([]float64{1.0 / math.Sqrt2})

	// Output:
	// variable[0.7071067811865475]
	// variable[0.7071067811865476]
	// [0.7071067811865476]
}

func ExampleSinT() {
	x := variable.New(math.Pi / 4)
	f := F.SinT{}

	fmt.Println(x)
	fmt.Println(f.Forward(x))
	fmt.Println(f.Backward(variable.OneLike(x)))

	// Output:
	// variable[0.7853981633974483]
	// [variable[0.7071067811865475]]
	// [variable[0.7071067811865476]]
}

func ExampleSin_double() {
	// p243
	x := variable.New(1.0)
	y := F.Sin(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	for i := 0; i < 10; i++ {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward()
		fmt.Println(x.Grad)
	}

	// Output:
	// variable[0.8414709848078965]
	// variable[0.5403023058681398]
	// variable[-0.8414709848078965]
	// variable[-0.5403023058681398]
	// variable[0.8414709848078965]
	// variable[0.5403023058681398]
	// variable[-0.8414709848078965]
	// variable[-0.5403023058681398]
	// variable[0.8414709848078965]
	// variable[0.5403023058681398]
	// variable[-0.8414709848078965]
	// variable[-0.5403023058681398]
}
