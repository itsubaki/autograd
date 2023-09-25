package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleSquare() {
	x := variable.New(1, 2, 3, 4, 5)
	y := variable.Square(x)
	y.Backward()

	fmt.Println(x)
	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[1 2 3 4 5]
	// variable[1 4 9 16 25]
	// variable[2 4 6 8 10]
}

func ExampleSquare_double() {
	x := variable.New(1, 2, 3, 4, 5)
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
	// variable[1 2 3 4 5]
	// variable[1 4 9 16 25]
	// variable[2 4 6 8 10]
	// variable[2 2 2 2 2]
	// variable[0 0 0 0 0]
}
