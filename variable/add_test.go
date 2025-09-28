package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleAddT() {
	a := variable.New(2, 3)
	b := variable.New(3, 4)
	f := variable.AddT{}

	fmt.Println(a)
	fmt.Println(b)
	fmt.Println(f.Forward(a, b))
	fmt.Println(f.Backward(variable.OneLike(a), variable.OneLike(b)))

	// Output:
	// variable[2]([2 3])
	// variable[2]([3 4])
	// [variable[2]([5 7])]
	// [variable[2]([1 1]) variable[2]([1 1])]
}

func ExampleAdd() {
	a := variable.New(2, 3)
	b := variable.New(3, 4)
	y := variable.Add(a, b)
	y.Backward()

	fmt.Println(a.Grad)
	fmt.Println(b.Grad)

	// Output:
	// variable[2]([1 1])
	// variable[2]([1 1])
}

func ExampleAddC() {
	x := variable.New(3)
	y := variable.AddC(10.0, x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable(13)
	// variable(1)
}

func ExampleAdd_broadcast() {
	// p305
	a := variable.New(1, 2, 3)
	b := variable.New(10)
	y := variable.Add(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad)
	fmt.Println(b.Grad)

	// Output:
	// variable[3]([11 12 13])
	// variable[3]([1 1 1])
	// variable(3)
}

func ExampleAdd_double() {
	a := variable.New(1, 2, 3)
	b := variable.New(10)

	y := variable.Add(a, b)
	y.Backward(variable.Opts{CreateGraph: true})
	fmt.Println(y)
	fmt.Println(a.Grad)
	fmt.Println(b.Grad)

	ga := a.Grad
	gb := b.Grad
	a.Cleargrad()
	b.Cleargrad()

	ga.Backward()
	gb.Backward()

	fmt.Println(a.Grad)
	fmt.Println(b.Grad)

	// Output:
	// variable[3]([11 12 13])
	// variable[3]([1 1 1])
	// variable(3)
	// <nil>
	// <nil>
}
