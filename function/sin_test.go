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
	fmt.Println(x)

	f := F.SinT{}
	fmt.Println(f.Forward(x.Data))
	fmt.Println(f.Backward(variable.OneLike(x)))

	// Output:
	// variable[0.7853981633974483]
	// [[0.7071067811865475]]
	// [variable[0.7071067811865476]]
}
