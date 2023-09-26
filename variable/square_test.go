package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleSquare() {
	x := variable.New(3.0)
	y := variable.Square(x)
	y.Backward()

	fmt.Println(x)
	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[3]
	// variable[9]
	// variable[6]
}

func ExampleSquare_double() {
	x := variable.New(3.0)
	y := variable.Square(x)
	y.Backward()

	fmt.Println(x)
	fmt.Println(y)
	fmt.Println(x.Grad)

	for i := 0; i < 2; i++ {
		gx := x.Grad
		x.Cleargrad()
		y.Cleargrad()
		gx.Backward()
		fmt.Println(x.Grad)
	}

	// Output:
	// variable[3]
	// variable[9]
	// variable[6]
	// variable[2]
	// variable[0]
}
