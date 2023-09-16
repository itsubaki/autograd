package function_test

import (
	"fmt"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
)

func ExampleSquare() {
	v := variable.New(1, 2, 3, 4, 5)
	y := F.Square(v)
	y.Backward()

	fmt.Println(v.Grad)

	// Output:
	// [2 4 6 8 10]

}

func ExampleSquareT() {
	v := variable.New(1, 2, 3, 4, 5)
	fmt.Println(v)

	f := F.SquareT{}
	fmt.Println(f.Forward(v.Data))

	v.Grad = f.Backward(variable.OneLike(v).Data)
	fmt.Println(v.Grad)

	// Output:
	// variable([1 2 3 4 5])
	// [1 4 9 16 25]
	// [2 4 6 8 10]

}
