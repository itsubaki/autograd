package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleMulT() {
	a := variable.New(3.0)
	b := variable.New(2.0)
	f := variable.MulT{}

	fmt.Println(a)
	fmt.Println(b)
	fmt.Println(f.Forward(a, b))
	fmt.Println(f.Backward(variable.OneLike(a), variable.OneLike(b)))

	// Output:
	// variable([3])
	// variable([2])
	// [variable([6])]
	// [variable([2]) variable([3])]
}

func ExampleMul() {
	// p139
	a := variable.New(3.0)
	b := variable.New(2.0)
	c := variable.New(1.0)
	y := variable.Add(variable.Mul(a, b), c)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable([7])
	// variable([2]) variable([3])
}

func ExampleMul_broadcast() {
	// p305
	a := variable.New(2, 2, 2, 2, 2)
	b := variable.New(3.0)
	y := variable.Mul(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable([6 6 6 6 6])
	// variable([3 3 3 3 3]) variable([10])
}
