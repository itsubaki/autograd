package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSub() {
	v := variable.New(2, 3)
	w := variable.New(3, 4)
	y := F.Sub(v, w)
	y.Backward()

	fmt.Println(y)
	fmt.Println(v.Grad)
	fmt.Println(w.Grad)

	// Output:
	// variable[-1 -1]
	// variable[1 1]
	// variable[-1 -1]
}

func ExampleSub_higher() {
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
