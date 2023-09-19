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
	v := variable.New(3.0)
	w := variable.New(2.0)
	f := F.MulT{}

	fmt.Println(v)
	fmt.Println(w)
	fmt.Println(f.Forward(v, w))
	fmt.Println(f.Backward(variable.OneLike(v), variable.OneLike(w)))

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
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	for i := 0; i < 2; i++ {
		ga, gb := a.Grad, b.Grad
		a.Cleargrad()
		b.Cleargrad()
		ga.Backward()
		gb.Backward()
		fmt.Println(a.Grad, b.Grad)
	}

	// Output:
	// variable[6]
	// variable[3] variable[2]
	// variable[1] variable[1]
	// <nil> <nil>
}
