package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleSub() {
	a := variable.New(3.0)
	b := variable.New(2.0)
	y := variable.Sub(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable[1]
	// variable[1] variable[-1]
}

func ExampleSubC() {
	x := variable.New(3.0)
	y := variable.SubC(10.0, x)
	y.Backward()

	fmt.Println(y)
	fmt.Println(x.Grad, y.Grad)

	// Output:
	// variable[7]
	// variable[-1] variable[1]
}

func ExampleSub_broadcast() {
	// p305
	a := variable.New(1, 2, 3, 4, 5)
	b := variable.New(1)
	y := variable.Sub(a, b)
	y.Backward()

	fmt.Println(y)
	fmt.Println(a.Grad, b.Grad)

	// Output:
	// variable[0 1 2 3 4]
	// variable[5] variable[-5]
}

func ExampleSub_double() {
	a := variable.New(3.0)
	b := variable.New(2.0)
	y := variable.Sub(a, b)
	y.Backward()

	ga, gb := a.Grad, b.Grad
	a.Cleargrad()
	b.Cleargrad()

	ga.Backward()
	fmt.Println(ga.Grad, gb.Grad)
	fmt.Println(y.Grad.Grad)            // ggy = gga
	fmt.Println(y.Grad.Grad == ga.Grad) // ggy is gga

	gb.Backward()
	fmt.Println(ga.Grad, gb.Grad)       // gb has neg creator
	fmt.Println(y.Grad.Grad)            // ggy = gga + Neg(ggb)
	fmt.Println(y.Grad.Grad == ga.Grad) // ggy is gga, gga is 0

	// Output:
	// variable[1] <nil>
	// variable[1]
	// true
	// variable[0] variable[1]
	// variable[0]
	// true
}

func ExampleSub_double_a() {
	a := variable.New(3.0)
	b := variable.New(2.0)
	y := variable.Sub(a, b)
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

func ExampleSub_double_b() {
	a := variable.New(3.0)
	b := variable.New(2.0)
	y := variable.Sub(a, b)
	y.Backward()

	gb := b.Grad
	a.Cleargrad()
	gb.Backward()
	fmt.Println(a.Grad, b.Grad) // gb has neg creator
	fmt.Println(b.Grad.Grad)    // 1
	fmt.Println(y.Grad.Grad)    // ggy = Neg(ggb)

	// Output:
	// <nil> variable[-1]
	// variable[1]
	// variable[-1]
}
