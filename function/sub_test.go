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
	y := F.Sub(a, b)
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
	y := F.Sub(a, b)
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
