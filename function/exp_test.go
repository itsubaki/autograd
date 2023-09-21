package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleExp() {
	v := variable.New(1, 2, 3, 4, 5)
	y := F.Exp(v)
	y.Backward()

	fmt.Println(v.Grad)

	// Output:
	// variable[2.718281828459045 7.38905609893065 20.085536923187668 54.598150033144236 148.4131591025766]
}

func ExampleExpT() {
	x := variable.New(1, 2, 3, 4, 5)
	f := F.ExpT{}

	fmt.Println(x)
	fmt.Println(f.Forward(x))
	fmt.Println(f.Backward(variable.OneLike(x)))

	// Output:
	// variable[1 2 3 4 5]
	// [variable[2.718281828459045 7.38905609893065 20.085536923187668 54.598150033144236 148.4131591025766]]
	// [variable[2.718281828459045 7.38905609893065 20.085536923187668 54.598150033144236 148.4131591025766]]
}

func ExampleExp_higher() {
	x := variable.New(2.0)
	y := F.Exp(x)
	y.Backward(variable.Opts{Retain: true})

	fmt.Println(y)
	fmt.Println(x.Grad)

	for i := 0; i < 10; i++ {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward(variable.Opts{Retain: true})
		fmt.Println(x.Grad)
	}

	// Output:
	// variable[7.38905609893065]
	// variable[7.38905609893065]
	// variable[7.38905609893065]
	// variable[7.38905609893065]
	// variable[7.38905609893065]
	// variable[7.38905609893065]
	// variable[7.38905609893065]
	// variable[7.38905609893065]
	// variable[7.38905609893065]
	// variable[7.38905609893065]
	// variable[7.38905609893065]
	// variable[7.38905609893065]
}
