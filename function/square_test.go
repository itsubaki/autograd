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

func ExampleSquareT() {
	x := variable.New(1, 2, 3, 4, 5)
	f := F.SquareT{}

	fmt.Println(x)
	fmt.Println(f.Forward(x.Data))
	fmt.Println(f.Backward(variable.OneLike(x)))

	// Output:
	// variable[1 2 3 4 5]
	// [[1 4 9 16 25]]
	// [variable[2 4 6 8 10]]
}
