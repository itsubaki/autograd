package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/autograd/vector"
)

func ExampleExp() {
	v := variable.New(1, 2, 3, 4, 5)
	y := F.Exp(v)
	y[0].Backward()

	fmt.Println(v.Grad)

	// Output:
	// [2.718281828459045 7.38905609893065 20.085536923187668 54.598150033144236 148.4131591025766]
}

func ExampleExpT() {
	v := variable.New(1, 2, 3, 4, 5)
	fmt.Println(v)

	f := F.ExpT{}
	fmt.Println(f.Forward(v.Data))
	fmt.Println(f.Backward(vector.OneLike(v.Data)))

	// Output:
	// variable([1 2 3 4 5])
	// [[2.718281828459045 7.38905609893065 20.085536923187668 54.598150033144236 148.4131591025766]]
	// [[2.718281828459045 7.38905609893065 20.085536923187668 54.598150033144236 148.4131591025766]]
}
