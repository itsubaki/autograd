package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSquare() {
	x := variable.New(1, 2, 3, 4, 5)
	y := F.Square(x)
	y.Backward()

	fmt.Println(x)
	fmt.Println(y)
	fmt.Println(x.Grad)

	// Output:
	// variable[1 2 3 4 5]
	// variable[1 4 9 16 25]
	// variable[2 4 6 8 10]
}

func ExampleSquare_higher() {
	x := variable.New(1, 2, 3, 4, 5)
	y := F.Square(x)
	y.Backward(variable.Opts{Retain: true})

	fmt.Println(x)
	fmt.Println(y)
	fmt.Println(x.Grad)

	for i := 0; i < 2; i++ {
		gx := x.Grad
		x.Cleargrad()
		gx.Backward(variable.Opts{Retain: true})
		fmt.Println(x.Grad)
	}

	// Output:
	// variable[1 2 3 4 5]
	// variable[1 4 9 16 25]
	// variable[2 4 6 8 10]
	// variable[2 2 2 2 2]
	// variable[0 0 0 0 0]
}
