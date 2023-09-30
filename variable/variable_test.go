package variable_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
)

func ExampleVariable() {
	v := variable.New(1, 2, 3, 4)
	fmt.Println(v)

	// Output:
	// variable([1 2 3 4])
}

func ExampleVariable_Name() {
	v := variable.New(1, 2, 3, 4)
	v.Name = "v"
	fmt.Println(v)

	// Output:
	// v([1 2 3 4])
}

func ExampleConst() {
	fmt.Println(variable.Const(1))

	// Output:
	// const([1])
}

func ExampleOneLike() {
	v := variable.New(1, 2, 3, 4)
	fmt.Println(variable.OneLike(v))

	// Output:
	// variable([1 1 1 1])
}

func ExampleVariable_Backward() {
	x := variable.New(1.0)
	x.Backward()
	fmt.Println(x.Grad)

	x.Cleargrad()
	x.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable([1])
	// variable([1])
}

func Example_gx() {
	fmt.Println(variable.Gx(nil, variable.New(1)))
	fmt.Println(variable.Gx(variable.New(1), variable.New(2)))

	// Output:
	// variable([1])
	// variable([3])
}
