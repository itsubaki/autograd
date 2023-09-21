package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleAdd() {
	a := variable.New(2, 3)
	b := variable.New(3, 4)
	y := F.Add(a, b)
	y.Backward()

	fmt.Println(a.Grad)
	fmt.Println(b.Grad)

	// Output:
	// variable[1 1]
	// variable[1 1]
}

func ExampleAddT() {
	a := variable.New(2, 3)
	b := variable.New(3, 4)
	f := F.AddT{}

	fmt.Println(a)
	fmt.Println(b)
	fmt.Println(f.Forward(a, b))
	fmt.Println(f.Backward(variable.OneLike(a), variable.OneLike(b)))

	// Output:
	// variable[2 3]
	// variable[3 4]
	// [variable[5 7]]
	// [variable[1 1] variable[1 1]]
}

func ExampleAdd_higher() {
	a := variable.New(2.0)
	b := variable.New(3.0)
	y := F.Add(a, b)
	y.Backward(variable.Opts{Retain: true})

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	for i := 0; i < 1; i++ {
		ga, gb := a.Grad, b.Grad
		a.Cleargrad()
		b.Cleargrad()
		ga.Backward(variable.Opts{Retain: true})
		gb.Backward(variable.Opts{Retain: true})
		fmt.Println(a.Grad, b.Grad)
	}

	// Output:
	// variable[5]
	// variable[1] variable[1]
	// <nil> <nil>
}
