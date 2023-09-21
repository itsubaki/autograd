package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleMul() {
	// p139
	a := variable.New(3.0)
	b := variable.New(2.0)
	c := variable.New(1.0)
	y := F.Add(F.Mul(a, b), c)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad)
	fmt.Println(b.Grad)

	// Output:
	// variable[7]
	// variable[2]
	// variable[3]
}

func ExampleMulT() {
	a := variable.New(3.0)
	b := variable.New(2.0)
	f := F.MulT{}

	fmt.Println(a)
	fmt.Println(b)
	fmt.Println(f.Forward(a, b))
	fmt.Println(f.Backward(variable.OneLike(a), variable.OneLike(b)))

	// Output:
	// variable[3]
	// variable[2]
	// [variable[6]]
	// [variable[2] variable[3]]
}

func ExampleMul_higher() {
	a := variable.New(2.0)
	b := variable.New(3.0)
	y := F.Mul(a, b)
	y.Backward(variable.Opts{Retain: true})

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	for i := 0; i < 2; i++ {
		ga, gb := a.Grad, b.Grad
		a.Cleargrad()
		b.Cleargrad()
		ga.Backward(variable.Opts{Retain: true})
		gb.Backward(variable.Opts{Retain: true})
		fmt.Println(a.Grad, b.Grad)
	}

	// Output:
	// variable[6]
	// variable[3] variable[2]
	// variable[1] variable[1]
	// <nil> <nil>
}
