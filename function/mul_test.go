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
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable[7]
	// variable[2] variable[3]
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

func ExampleMul_double() {
	a := variable.New(2.0)
	b := variable.New(3.0)
	y := F.Mul(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	ga, gb := a.Grad, b.Grad
	a.Cleargrad()
	b.Cleargrad()
	ga.Backward()
	gb.Backward()
	fmt.Println(a.Grad, b.Grad)
	fmt.Println(y.Grad, y.Grad.Grad)

	// Output:
	// variable[6]
	// variable[3] variable[2]
	// variable[1] variable[1]
	// variable[1] variable[5]
}

func ExampleMul_double_a() {
	a := variable.New(2.0)
	b := variable.New(3.0)
	y := F.Mul(a, b)
	y.Backward()

	ga := a.Grad
	b.Cleargrad()
	ga.Backward()
	fmt.Println(a.Grad, b.Grad)
	fmt.Println(y.Grad.Grad)

	// Output:
	// variable[3] variable[1]
	// variable[3]
}

func ExampleMul_double_b() {
	a := variable.New(2.0)
	b := variable.New(3.0)
	y := F.Mul(a, b)
	y.Backward()

	gb := b.Grad
	a.Cleargrad()
	gb.Backward()
	fmt.Println(a.Grad, b.Grad)
	fmt.Println(y.Grad.Grad)

	// Output:
	// variable[1] variable[2]
	// variable[2]
}
