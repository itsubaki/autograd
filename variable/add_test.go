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
	// variable[2 3]
	// variable[3 4]
	// [variable[5 7]]
	// [variable[1 1] variable[1 1]]
}

func ExampleAdd() {
	a := variable.New(2, 3)
	b := variable.New(3, 4)
	y := variable.Add(a, b)
	y.Backward()

	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable[1 1] variable[1 1]
}

func ExampleAddC() {
	x := variable.New(1, 2, 3, 4, 5)
	y := variable.AddC(10.0, x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[11 12 13 14 15]
	// variable[1 1 1 1 1]
}

func ExampleAdd_double() {
	a := variable.New(3.0)
	b := variable.New(2.0)
	y := variable.Add(a, b)
	y.Backward()

	ga, gb := a.Grad, b.Grad
	a.Cleargrad()
	b.Cleargrad()

	ga.Backward()
	fmt.Println(y.Grad.Grad, ga.Grad, gb.Grad)
	fmt.Println(y.Grad.Grad == ga.Grad, y.Grad.Grad == gb.Grad)

	gb.Backward()
	fmt.Println(y.Grad.Grad, ga.Grad, gb.Grad)
	fmt.Println(y.Grad.Grad == ga.Grad, y.Grad.Grad == gb.Grad)

	// Output:
	// variable[1] variable[1] variable[1]
	// true true
	// variable[1] variable[1] variable[1]
	// true true
}

func ExampleAdd_double_a() {
	a := variable.New(3.0)
	b := variable.New(2.0)
	y := variable.Add(a, b)
	y.Backward()

	ga := a.Grad
	b.Cleargrad()
	ga.Backward()
	fmt.Println(a.Grad, b.Grad)             // ga has no creator
	fmt.Println(a.Grad.Grad)                // 1
	fmt.Println(y.Grad.Grad == a.Grad.Grad) // ggy = gga

	// Output:
	// variable[1] <nil>
	// variable[1]
	// true
}

func ExampleAdd_double_b() {
	a := variable.New(3.0)
	b := variable.New(2.0)
	y := variable.Add(a, b)
	y.Backward()

	gb := b.Grad
	a.Cleargrad()
	gb.Backward()
	fmt.Println(a.Grad, b.Grad)             // gb has no creator
	fmt.Println(b.Grad.Grad)                // 1
	fmt.Println(y.Grad.Grad == b.Grad.Grad) // ggy = ggb

	// Output:
	// <nil> variable[1]
	// variable[1]
	// true
}
