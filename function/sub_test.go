package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSub() {
	a := variable.New(2, 3)
	b := variable.New(3, 4)
	y := F.Sub(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad)
	fmt.Println(b.Grad)

	// Output:
	// variable[-1 -1]
	// variable[1 1]
	// variable[-1 -1]
}

func ExampleSubC() {
	x := variable.New(1, 2, 3, 4, 5)
	y := F.SubC(10.0, x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[9 8 7 6 5]
	// variable[-1 -1 -1 -1 -1]
}

func ExampleSub_double() {
	a := variable.New(3.0)
	b := variable.New(2.0)
	y := F.Sub(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	for i := 0; i < 1; i++ {
		ga, gb := a.Grad, b.Grad
		a.Cleargrad()
		b.Cleargrad()
		ga.Backward()
		gb.Backward()
		fmt.Println(a.Grad, b.Grad)
	}

	// Output:
	// variable[1]
	// variable[1] variable[-1]
	// <nil> <nil>
}
