package function_test

import (
	"fmt"
	"math"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleCos() {
	// p198
	x := variable.New(math.Pi / 4)
	y := F.Cos(x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)
	fmt.Println([]float64{1.0 / math.Sqrt2})

	// Output:
	// variable[0.7071067811865476]
	// variable[-0.7071067811865475]
	// [0.7071067811865476]
}

func ExampleCosT() {
	x := variable.New(math.Pi / 4)
	f := F.CosT{}

	fmt.Println(x)
	fmt.Println(f.Forward(x))
	fmt.Println(f.Backward(variable.OneLike(x)))

	// Output:
	// variable[0.7853981633974483]
	// [variable[0.7071067811865476]]
	// [variable[-0.7071067811865475]]
}
